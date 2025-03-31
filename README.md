# ReAL: Recall-Oriented Adaptive Learning for PLM-aided Query Expansion in Open-Domain Question Answering

## TLDR

ReAL introduces a novel approach to enhance query expansion (QE) in open-domain question answering (ODQA) systems, especially those utilizing a retriever-reader architecture. Traditional QE methods that use pre-trained language models (PLMs) often treat all expanded terms equally, which can lead to suboptimal retrieval accuracy. ReAL addresses this issue by dynamically adjusting the importance of each expanded term based on its relevance. Through iterative refinement of term weights using a similarity-based model and custom loss functions, ReAL improves the separation of relevant and irrelevant documents. Our experiments across multiple datasets and QE methods demonstrate that ReAL consistently boosts retrieval performance, making it a promising enhancement for ODQA systems.

## Retriever

For retrieval, ReAL uses pyserini for BM25-based document retrieval. Pyserini handles indexing and searching over large text corpora such as Wikipedia, which can be downloaded via the `download_data_NQTQA.py` script.

Additionally, ReAL utilizes a modified version of pyserini called `weightedsearcher`. This package extends pyserini by incorporating token weight vectors that are dynamically updated during retrieval. The key components of `weightedsearcher` include:

- **weightedsearcher.py**: A modified version of pyserini’s `SimpleSearcher` for performing weighted retrieval.
- **weight_optimization_rank_and_contrast.py**: Implements the token weight optimization algorithm.
- **classifier_\* modules**: Use different classifiers for relevance judgment and document classification.

The retrieval process with ReAL is executed using `weighted_update_search.py`, which integrates the `weightedsearcher` package to apply the Recall-Oriented Adaptive Learning (ReAL) method.

For evaluating retrieval performance, we use the `wepyserini` package to calculate Hit@k scores. The `cal_hit_multi.py` script allows for efficient evaluation of retrieval results.

## Reader

ReAL integrates with an extractive reader based on the FiD (Fusion-in-Decoder) model, which is effective for answer extraction in ODQA systems. To use the FiD reader, first download the relevant files from the [FiD GitHub repository](https://github.com/facebookresearch/FiD) and place them in the `reader` directory. You can then trigger the answer extraction process using the `eval_update.sh` script.

The reader expects data in the following format, where each entry is a dictionary containing:
  - `id`: An optional identifier for the example.
  - `question`: The question text.
  - `target`: The correct answer used for model training (if not provided, a random answer is sampled from the `answers` list).
  - `answers`: A list of possible answers for evaluation and training.
  - `ctxs`: A list of passages containing relevant context for the question. Each passage includes:
    - `title`: The title of the article.
    - `text`: The text content of the passage.

Example entry:

```json
{
  "id": "0",
  "question": "What element did Marie Curie name after her native land?",
  "target": "Polonium",
  "answers": ["Polonium", "Po (chemical element)", "Po"],
  "ctxs": [
    {
      "title": "Marie Curie",
      "text": "Marie Curie named the element polonium after her native Poland..."
    }
  ]
}
```
The `qa_eval` package is used for automatic evaluation of the end-to-end question answering results. It integrates large models for evaluating the relevance of the answers. The evaluation scripts are triggered by run_update.sh, which calls the relevant metrics and scoring functions.

