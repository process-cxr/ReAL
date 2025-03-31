#%%
import os
from typing import Dict, List, Optional, Union

from pyserini.fusion import FusionMethod, reciprocal_rank_fusion
from pyserini.index import Document, IndexReader
from pyserini.pyclass import autoclass, JFloat, JArrayList, JHashMap
from pyserini.search import JQuery, JQueryGenerator
from pyserini.trectools import TrecRun
from pyserini.util import download_prebuilt_index, get_sparse_indexes_info
import json

# Set the cache directory
os.environ["PYSERINI_CACHE"] = "bm25_cache/"

# Use pyserini's autoclass to define necessary Java classes
JWeightedSearcher = autoclass('io.anserini.search.WeightedSearcher')
JScoredDoc = autoclass('io.anserini.search.ScoredDoc')
JDocument = autoclass('org.apache.lucene.document.Document')

# Define a Python class to interact with the Java WeightedSearcher
class WeightedSearcherWrapper:
    def __init__(self, index_dir: str):
        self.searcher = JWeightedSearcher(index_dir)

    def search(self, query: str, top_n: int):
        results = self.searcher.search(query, top_n)
        final_results = []
        for rank, result in enumerate(results):
            docid = result.docid
            doc = self.searcher.doc(docid)
            doc_contents = self._get_document_contents(doc)
            doc_dict = {
                'id': docid,
                'rank': rank + 1,
                'text': doc_contents,
                'weBM25_score': result.score,
            }
            final_results.append(doc_dict)

        # Get the initial term weights using the searcher
        initial_weights = self._convert_java_map_to_dict(self.searcher.getTermWeights())
        token_vec = {k: float(v) for k, v in initial_weights.items()}

        return final_results, token_vec, results

    def update_weight_vector_and_search(self, token_vec_new: Dict[str, float], query: str, top_n: int):
        # Convert Python dictionary to Java HashMap
        java_token_vec_new = JHashMap()
        for token, weight in token_vec_new.items():
            java_token_vec_new.put(token, float(weight))

        # Update the weight vector in the searcher
        self.searcher.updateWeightVector(java_token_vec_new)

        # Perform a new search with the updated weight vector
        results = self.searcher.search(query, top_n)
        final_results = []
        for rank, result in enumerate(results):
            docid = result.docid
            doc = self.searcher.doc(docid)
            doc_contents = self._get_document_contents(doc)
            doc_dict = {
                'id': docid,
                'rank': rank + 1,
                'text': doc_contents,
                'weBM25_score': result.score,
            }
            final_results.append(doc_dict)

        return final_results, results

    def get_token_scores_for_results(self, query: str, results: JScoredDoc):        
        # Call the Java method getTokenScoresForResults
        token_scores = self.searcher.getTokenScoresForResults(query, results)
        
        # Prepare the final output structure
        token_scores_results = []
        for result in results:
            docid = result.docid
            doc_token_scores = self._convert_java_map_to_dict(token_scores.get(docid))
            result_dict = {
                "id": docid,
                "token_score": doc_token_scores,
                "doc_score": result.score,
            }
            token_scores_results.append(result_dict)

        return token_scores_results

    def get_total_num_docs(self):
        return self.searcher.get_total_num_docs()

    def close(self):
        self.searcher.close()

    def _get_document_contents(self, doc: JDocument) -> str:
        raw_json = doc.get('raw')
        if raw_json:
            doc_dict = json.loads(raw_json)
            return doc_dict.get('contents', '')
        return ''

    def _convert_java_map_to_dict(self, java_map) -> Dict[str, float]:
        # Convert a Java HashMap to a Python dictionary
        py_dict = {}
        it = java_map.entrySet().iterator()
        while it.hasNext():
            entry = it.next()
            py_dict[entry.getKey()] = entry.getValue()
        return py_dict


