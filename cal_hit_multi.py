#%%
from wepyserini.retriever_utils import load_passages, validate, save_results
import pickle
import os
import csv 

#%%

def load_data_with_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def process_and_save_retrieval_results(top_docs, dataset_name, questions, question_answers, all_passages, num_threads, match_type, output_dir, output_no_text=False):
    recall_outfile = os.path.join(output_dir, 'recall_at_k.csv')
    result_outfile = os.path.join(output_dir, 'results.json')
    
    questions_doc_hits = validate(
        dataset_name,
        all_passages,
        question_answers,
        top_docs,
        num_threads,
        match_type,
        recall_outfile,
        use_wandb=False
    )
    
    save_results(
        all_passages,
        questions,
        question_answers,
        top_docs,
        questions_doc_hits,
        result_outfile,
        output_no_text=output_no_text
    )
    
    return questions_doc_hits

#%%
if __name__=='__main__':
    
    dataset_name = 'nq'
    num_threads = 10
    match_type = 'regex'  # match_type = "regex" if "curated" in dataset_name else args.match
    output_no_text = False
    ctx_file = 'data/downloads/data/wikipedia_split/psgs_w100.tsv'

    input_file_path = 'data/downloads/data/retriever/qas/nq-dev.csv'
    with open(input_file_path,'r') as file:
        query_data = csv.reader(file, delimiter='\t')
        questions, question_answers = zip(*[(item[0], eval(item[1])) for item in query_data])
        questions = questions
        question_answers = question_answers
    
    all_passages = load_passages(ctx_file)

    results_list =  [""]
    update_output_dir = [f"""output/{str(item)}""" for item in results_list]
    update_top_docs_list_pkl_path = 'temp_pkl/results/update_top_docs_list.pkl'
    
    update_top_docs_list = load_data_with_pickle(update_top_docs_list_pkl_path)


    update_top_docs_list = zip(*update_top_docs_list)
    for path, docs in zip(update_output_dir, update_top_docs_list):
        os.makedirs(path, exist_ok=True)
        update_questions_doc_hits = process_and_save_retrieval_results(
            docs,
            dataset_name,
            questions,
            question_answers,
            all_passages,
            num_threads,
            match_type,
            path,
            output_no_text=output_no_text
        )
    print('Validation End!')
