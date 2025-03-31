#!/bin/bash

answer_txt="data/downloads/data/retriever/qas/nq-dev.csv"
prompt_txt="qa_eval/prompts/eval-v0.2-few-shot_chat.txt"

items=("gar_qe")
n_contexts=(100 50 20 1)

for n_context in "${n_contexts[@]}"; do
  for item in "${items[@]}"; do
    base_out_path="qa_eval/nq-dev/EM@${n_context}/nq-dev-${item}_update/"
    prediction_txt="fid-reader/nq-dev/EM@${n_context}/nq-dev-${item}_update/final_output.txt"

    echo "Executing: CUDA_VISIBLE_DEVICES="0,1,2,3" python qa_eval/llms_eval.py --base_out_path $base_out_path --answer_txt  $answer_txt --prediction_txt  $prediction_txt --prompt_txt  $prompt_txt"

    CUDA_VISIBLE_DEVICES="0,1,2,3" python qa_eval/llms_eval.py \
    --base_out_path $base_out_path \
    --answer_txt  $answer_txt \
    --prediction_txt  $prediction_txt \
    --prompt_txt  $prompt_txt
  done
done