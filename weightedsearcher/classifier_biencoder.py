import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class BiEncoderRanker:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)

    def score_documents(self, query: str, doc_dicts: list) -> list:
        # 编码 query 和所有文档
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        doc_texts = [doc['text'] for doc in doc_dicts]
        doc_embeddings = self.model.encode(doc_texts, normalize_embeddings=True)

        # 相似度计算（归一化后的点积 = 余弦相似度）
        similarities = np.dot(doc_embeddings, query_embedding)

        # 整理输出
        re_doc_dicts = []
        for doc_dict, score in zip(doc_dicts, similarities):
            re_doc_dicts.append({
                'id': doc_dict['id'],
                'ranker_score': float(score),
                'bm25_score': doc_dict.get('doc_score', 0)
            })

        # 按 ranker_score 排序
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
            print("Cleared CUDA cache.")
