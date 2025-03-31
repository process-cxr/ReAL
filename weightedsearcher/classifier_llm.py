# %%
import os
from vllm import LLM, SamplingParams
import torch
import json
import ray


class LLMClassifier:
    def __init__(self, llm_model_path, device_ids="0,1,2,3"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        self.llm_base = LLM(llm_model_path, tensor_parallel_size=len(device_ids.split(',')),enforce_eager=True)
        self.device_ids = device_ids
        
    def score_documents(self, query, doc_dicts, max_tokens=3, top_n=100):
        
        relevant = []
        non_relevant = []
        for doc in doc_dicts:
            judgment = self.llm_judge_relevance(query, doc['text'], max_tokens)
            if "yes" in judgment.lower():
                relevant.append((doc['id'],judgment))
            else:
                non_relevant.append((doc['id'],judgment))
        
        relevant_docs = [{
                    'id': id,
                    'ranker_score': top_n - idx,
                    'judgment': judgment,
                }
                         for idx, (id, judgment) in enumerate(relevant)]
        non_relevant_docs = [{
                    'id': id,
                    'ranker_score': top_n - len(relevant_docs) - idx,
                    'judgment': judgment,
                }
                         for idx, (id, judgment) in enumerate(non_relevant)]

        sorted_docs = relevant_docs + non_relevant_docs

        return sorted_docs
    
    def llm_judge_relevance(self, query, doc, max_tokens=3):
        instruction ="""You are an expert in analyzing text. Please read the following document.
        
        Document: {doc}

        If the document above contains the answer of question "{query}", respond "Yes". If the document does not contain the answer of the question "{query}", respond "No".

        Firtly give you respond and then give the answer of question "{query}".""".format(query=query,doc=doc)
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        output = self.llm_base.generate(instruction, sampling_params)
        answer = output[0].outputs[0].text.replace('\n', ' ').strip()
        return answer
    
    def close(self):
        for device_id in self.device_ids:
            torch.cuda.set_device(f"cuda:{device_id}")
            torch.cuda.empty_cache()
        print("Cleared CUDA cache for devices:", self.device_ids)

