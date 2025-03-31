#!/bin/bash

model_path="/data/cxr/cxrmain/proj_QE/FiD/pretrained_models/tqa_reader_large"
per_gpu_batch_size=1
write_results="--write_results"

# 参数列表
items=("gar_qe")
n_contexts=(1 20 50 100)

# 循环遍历参数
for item in "${items[@]}"; do
  for n_context in "${n_contexts[@]}"; do
    checkpoint_dir="/data/cxr/cxrmain/proj_QE/FiD/result_trivia/crossrank_and_contrast/EM@${n_context}"
    eval_data="/data/cxr/cxrmain/proj_QE/FiD/data/trivia/trivia-test-${item}_crossrank_and_contrast_update.json"
    name="trivia-test-${item}_update"

    echo "Executing: CUDA_VISIBLE_DEVICES=6 python test_reader.py --model_path $model_path --eval_data $eval_data --per_gpu_batch_size $per_gpu_batch_size --n_context $n_context --name $name --checkpoint_dir $checkpoint_dir $write_results"

    CUDA_VISIBLE_DEVICES=6 python test_reader.py \
      --model_path $model_path \
      --eval_data $eval_data \
      --per_gpu_batch_size $per_gpu_batch_size \
      --n_context $n_context \
      --name $name \
      --checkpoint_dir $checkpoint_dir \
      $write_results
  done
done