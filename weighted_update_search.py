#%%
from weightedsearcher.weightedseacher import WeightedSearcherWrapper
from weightedsearcher.classifier_crossencoder import Classifier
from weightedsearcher.weight_optimization_rank_and_contrast import DistributionForUpdate
import csv
import os
import tqdm
import pickle
import logging
import json
#%%

def weight_optimize(classifier_res, vec_dict, bm25_res, lr, max_iterations, top_num, bottom_num, mask):
    rank_loss = DistributionForUpdate(classifier_res, vec_dict,bm25_res,lr, max_iterations,top_num=top_num,bottom_num=bottom_num,mask=mask) 
    rank_loss.iterate_process()
    new_vec_dict, log_list = rank_loss.save_updated_weights()
    return new_vec_dict, log_list

def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_with_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

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



#%%
if __name__ == "__main__":
    log_path = 'log/record.log'
    logger = setup_logger(log_path)
    logger.info("Validation Start!")

    ####  主要运行参数
    index_dir = "bm25_cache/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.7b58c08da992b2ea7e96667f0b176651"
    model_path = "model/cross_encoder/"
    device = 'cuda:0'
    input_file_path = 'downloads/data/retriever/qas/nq-dev.csv'
    qe_file_path = 'qe_generation/nq-dev-qe.txt'
    with open(input_file_path, 'r') as q_file, open(qe_file_path, 'r') as qe_file:
        query_data = csv.reader(q_file, delimiter='\t') 
        qe_data = csv.reader(qe_file, delimiter='\t' , quoting= csv.QUOTE_NONE) 
        questions, question_answers = zip(*[(query[0]+" "+ qe[1], eval(query[1])) for query, qe in zip(query_data, qe_data)])  

    logger.info(f"""questions number: {len(questions)}""")
    lr_list = [0.5,0.1,0.05]
    contrast_num_list = [(10,10)]
    max_iterations = 1000
    top_n = 100
    rank_num = 30
    top_docs =[]
    classifier_top_docs = []
    update_top_docs_list = []
    filter_flag = False
    update_weight_dict = {}
    use_token_score_cache = True
        
    if use_token_score_cache:        
        token_score_pkl_path = 'temp_pkl/token_score.pkl'
        token_score = load_data_with_pickle(token_score_pkl_path)
    else:
        token_score = []
    cross_classifier = Classifier(model_path,device)
    for idx, question in tqdm.tqdm(enumerate(questions,1)):
        searcher = WeightedSearcherWrapper(index_dir)
        # 首次进行BM25索引
        bm25_results, token_vec, java_bm25_results = searcher.search(question, top_n)
        logger.info(f"""token_vec_length of question {idx}: {len(token_vec)}""")
        if use_token_score_cache:
            token_scores_results = token_score[idx-1]
        else:
            token_scores_results = searcher.get_token_scores_for_results(question, java_bm25_results)
            token_score.append(token_scores_results)
        for item in token_scores_results:
                token_scores = item['token_score']
                token_score_sum = sum(token_scores.values())
                if abs(item['doc_score'] - token_score_sum) > 1e-1:  logger.info(f"""Warning, doc_score {item['doc_score']} is not equal to token_score_sum {token_score_sum} for question """)
        
        logger.info(f"""Token_scores_results[0] of question {idx}: {token_scores_results[0]}""")
        docs_id = [item['id']  for item in bm25_results]
        docs_score = [item['doc_score']  for item in bm25_results]
        top_docs.append((docs_id, docs_score))

        ranked_docs = cross_classifier.score_documents(question, bm25_results)
        sorted_ranked_docs = cross_classifier.forward_relevant(ranked_docs, rank_num)

        classifier_docs_id = [item['id']  for item in sorted_ranked_docs]
        classifier_docs_score = [item['classifier_score']  for item in sorted_ranked_docs]
        classifier_top_docs.append((classifier_docs_id, classifier_docs_score))
            
        update_top_docs = []
        idx_weight_dict = {}
        for lr in lr_list:
            top_num = 10
            bottom_num = 10
            # 权重进行更新
            mask = len([item for item in sorted_ranked_docs if "yes" in item["judgment"].lower()])
            new_vec_dict, log_list = weight_optimize(sorted_ranked_docs, token_vec, token_scores_results, lr, max_iterations, top_num, bottom_num, mask)
            idx_weight_dict[str(lr)] = new_vec_dict
            logger.info(f"""loss iteration of question {idx}:""")
            for log_text in log_list:
                logger.info(log_text)
            update_results, update_java_results = searcher.update_weight_vector_and_search(new_vec_dict, question, top_n)
            update_docs_id = [item['id']  for item in update_results]
            update_docs_score = [item['doc_score']  for item in update_results]
            update_top_docs.append((update_docs_id, update_docs_score))
        update_weight_dict[str(idx)] = idx_weight_dict
        update_top_docs_list.append(update_top_docs)
        searcher.close()
    
    if use_token_score_cache:
        token_score_pkl_path = 'temp_pkl/'
        save_data_with_pickle(token_score, os.path.join(token_score_pkl_path,'token_score.pkl'))

    
    output_pkl_path = 'temp_pkl/results/'
    os.makedirs(output_pkl_path, exist_ok=True)
    save_data_with_pickle(update_top_docs_list, os.path.join(output_pkl_path,'update_top_docs_list.pkl'))
    save_data_with_pickle(update_weight_dict, os.path.join(output_pkl_path,'update_weight_dict.pkl'))

    logger.info("First Retrieval and Updated Retrieval Completed!")

