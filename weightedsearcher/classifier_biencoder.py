import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

class biClassifier:
    def __init__(self, model_path, device):
        self.device = device

        # 初始化模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def score_documents(self, query: str, doc_dicts: list) -> list:
        # 准备 query 和文档对
        pairs = [(query, doc['text']) for doc in doc_dicts]

        # 对 pairs 进行 tokenization
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)

        # 使用模型计算得分
        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1, ).float()
        
        # 将 logits 转为 tensor，并使用 softmax 计算归一化分数
        scores_tensor = logits.to(self.device)
        scores_softmax = softmax(scores_tensor, dim=0).cpu().numpy()

        # 整理文档分数
        re_doc_dicts = []
        for doc_dict, score, softmax_score in zip(doc_dicts, scores_tensor.cpu().numpy(), scores_softmax):
            re_doc_dict = {
                'id': doc_dict['id'],
                'ranker_score': score,
                'bm25_score': doc_dict.get('doc_score', 0)  # 假设每个文档都有 bm25_score
            }
            re_doc_dicts.append(re_doc_dict)

        # 按 ranker_score 排序
        re = sorted(re_doc_dicts, key=lambda x: x['ranker_score'], reverse=True)
        return re

    def forward_relevant(self, re_doc_dicts: list, num: int = 10) -> list:
        relevant = re_doc_dicts[:num]
        non_relevant = re_doc_dicts[num:]
        idx = len(re_doc_dicts)

        # 根据 bm25_score 对相关文档和非相关文档分别排序
        relevant_sorted = sorted(relevant, key=lambda x: x['bm25_score'], reverse=True)
        non_relevant_sorted = sorted(non_relevant, key=lambda x: x['bm25_score'], reverse=True)

        # 整理相关文档和非相关文档
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

        # 返回合并后的结果
        return relevant_doc + non_relevant_doc

    def close(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            print("Cleared CUDA cache")
