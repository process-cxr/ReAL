from sentence_transformers import CrossEncoder
from torch.nn.functional import softmax
import torch

class Classifier:
    def __init__(self, model_path,device):
        self.model_path = model_path
        self.device = device
        self.cross_encoder = CrossEncoder(self.model_path, device=self.device)  
    
    def score_documents(self, query, doc_dicts):
        pairs = [(query, doc['text']) for doc in doc_dicts]
        scores = self.cross_encoder.predict(pairs)
        scores_tensor = torch.tensor(scores).to(self.device)  
        scores_softmax = softmax(scores_tensor, dim=0).cpu().numpy() 
        re_doc_dicts =[]
        for doc_dict, socre, softmax_score in zip(doc_dicts, scores, scores_softmax):
            re_doc_dict = {}
            re_doc_dict['id'] = doc_dict['id']
            re_doc_dict['ranker_score'] = socre
            re_doc_dict['bm25_score'] = doc_dict['doc_score']
            doc_dict['ranker_score'] = socre
            re_doc_dicts.append(re_doc_dict)
        re = sorted(re_doc_dicts,key=lambda x : x['ranker_score'], reverse=True)
        return re
    
    
    def forward_relevant(self, re_doc_dicts, num=10):
        relevant = re_doc_dicts[0:num]
        non_relevant = re_doc_dicts[num:]
        idx = len(re_doc_dicts)

        relevant_sorted = sorted(relevant, key=lambda x: x['bm25_score'], reverse=True)
        non_relevant_sorted = sorted(non_relevant, key=lambda x: x['bm25_score'], reverse=True)
        
        relevant_doc = [
            {
                    "id": item['id'],
                    "ranker_score": idx-i,
                    "judgment": "Yes",
                    "orignal_judgment": f"bm25_score: {item['bm25_score']}, cranker_score: {item['ranker_score']}"
                }
            for (i, item) in enumerate(relevant_sorted)
        ]
        non_relevant_doc = [  
            {
                    "id": item['id'],
                    "ranker_score": idx-len(relevant_doc)-i,
                    "judgment": "No",
                    "orignal_judgment": f"bm25_score: {item['bm25_score']}, cranker_score: {item['ranker_score']}"
                }
            for (i, item) in enumerate(non_relevant_sorted)
        ]
        res = relevant_doc + non_relevant_doc
        return res
    
    
    def close(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")
