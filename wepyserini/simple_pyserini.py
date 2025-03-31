#%%
import logging
import sys
from typing import List, Tuple
from pyserini.search import SimpleSearcher
from retriever_utils import load_passages, validate, save_results
import os
import pickle
import csv

os.environ["PYSERINI_CACHE"] = "bm25_cache/"

def setup_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

class SparseRetriever:
    def __init__(self, index_name, log_path,  num_threads=1):
        self.searcher = SimpleSearcher.from_prebuilt_index(index_name)
        self.num_threads = num_threads
        self.dedup = False
        
    def get_top_docs(
        self, questions: List[str], top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        qids = [str(x) for x in range(len(questions))]
        if self.dedup:
            dedup_q = {}
            dedup_map = {}
            for qid, question in zip(qids, questions):
                if question not in dedup_q:
                    dedup_q[question] = qid
                else:
                    dedup_map[qid] = dedup_q[question]
            dedup_questions = []
            dedup_qids = []
            for question in dedup_q:
                qid = dedup_q[question]
                dedup_questions.append(question)
                dedup_qids.append(qid)
            hits = self.searcher.batch_search(queries=dedup_questions, qids=dedup_qids, k=top_docs, threads=self.num_threads)
            for qid in dedup_map:
                hits[qid] = hits[dedup_map[qid]]
        else:
            hits = self.searcher.batch_search(queries=questions, qids=qids, k=top_docs, threads=self.num_threads)
        results = []
        for qid in qids:
            example_hits = hits[qid]
            example_top_docs = [hit.docid for hit in example_hits]
            example_scores = [hit.score for hit in example_hits]
            results.append((example_top_docs, example_scores))
        return results


if __name__ == "__main__":
    
    ####  主要运行参数
    log_path = 'log/record.log'
    top_docs_pkl_path = ''
    input_file_path = ''
    qe_file_path = ''
    with open(input_file_path, 'r') as q_file, open(qe_file_path, 'r') as qe_file:
        query_data = csv.reader(q_file, delimiter='\t') 
        qe_data = csv.reader(qe_file, delimiter='\t' , quoting= csv.QUOTE_NONE)
        questions, question_answers = zip(*[(query[0]+" "+ qe[1], eval(query[1])) for query, qe in zip(query_data, qe_data)]) 
        questions, question_answers = zip(*[(query[0], eval(query[1])) for query, qe in zip(query_data, qe_data)])  

        
    print(questions)

    index_name = "wikipedia-dpr"  
    retriever = SparseRetriever(index_name,log_path)
    top_docs_list = retriever.get_top_docs(questions)
    os.makedirs(top_docs_pkl_path, exist_ok=True)
    save_data_with_pickle(top_docs_list, os.path.join(top_docs_pkl_path,'top_docs_1it_stable.pkl'))
