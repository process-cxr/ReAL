import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

class biClassifier:
    def __init__(self, model_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def score_documents(self, query: str, doc_dicts: list) -> list:
        pairs = [(query, doc['text']) for doc in doc_dicts]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1, ).float()
        scores_tensor = logits.to(self.device)
        scores_softmax = softmax(scores_tensor, dim=0).cpu().numpy()
        re_doc_dicts = []
        for doc_dict, score, softmax_score in zip(doc_dicts, scores_tensor.cpu().numpy(), scores_softmax):
            re_doc_dict = {
                'id': doc_dict['id'],
                'ranker_score': score,
                'bm25_score': doc_dict.get('doc_score', 0)
            }
            re_doc_dicts.append(re_doc_dict)

        re = sorted(re_doc_dicts, key=lambda x: x['ranker_score'], reverse=True)
        return re

    def forward_relevant(self, re_doc_dicts: list, num: int = 10) -> list:
        relevant = re_doc_dicts[:num]
        non_relevant = re_doc_dicts[num:]
        idx = len(re_doc_dicts)

        relevant_sorted = sorted(relevant, key=lambda x: x['bm25_score'], reverse=True)
        non_relevant_sorted = sorted(non_relevant, key=lambda x: x['bm25_score'], reverse=True)
        relevant_doc = [
            {
                "id": item['id'],
                "ranker_score": idx - i,
                "judgment": "Yes",
                "orignal_judgment": f"bm25_score: {item['bm25_score']}, biencoder_score: {item['ranker_score']}"
            }
            for i, item in enumerate(relevant_sorted)
        ]

        non_relevant_doc = [
            {
                "id": item['id'],
                "ranker_score": idx - len(relevant_doc) - i,
                "judgment": "No",
                "orignal_judgment": f"bm25_score: {item['bm25_score']}, biencoder_score: {item['ranker_score']}"
            }
            for i, item in enumerate(non_relevant_sorted)
        ]

        return relevant_doc + non_relevant_doc

    def close(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")
