#!/bin/bash

model_path="/pretrained_models/tqa_reader_large"
per_gpu_batch_size=1
write_results="--write_results"

# 参数列表
items=("gar_qe")
n_contexts=(1 20 50 100)

# 循环遍历参数
for item in "${items[@]}"; do
  for n_context in "${n_contexts[@]}"; do
    checkpoint_dir="data/nq-dev/EM@${n_context}"
    eval_data="data/nq-dev/nq-dev-${item}.json"
    name="nq-dev-${item}_update"

    echo "Executing: CUDA_VISIBLE_DEVICES=0 python test_reader.py --model_path $model_path --eval_data $eval_data --per_gpu_batch_size $per_gpu_batch_size --n_context $n_context --name $name --checkpoint_dir $checkpoint_dir $write_results"

    CUDA_VISIBLE_DEVICES=0 python test_reader.py \
      --model_path $model_path \
      --eval_data $eval_data \
      --per_gpu_batch_size $per_gpu_batch_size \
      --n_context $n_context \
      --name $name \
      --checkpoint_dir $checkpoint_dir \
      $write_results
  done
done