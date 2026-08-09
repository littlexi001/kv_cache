#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval
output_root="outputs/xsum_10m_selected_order_ablation_v1"
mkdir -p "${output_root}"

pids=()
for rank in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${rank}" \
    /home/fdong/miniconda3/envs/moe/bin/python \
    src/evaluate_xsum_10m_selected_order_ablation.py \
    --data_dir data/xsum_10m_continuation_memory_v1 \
    --selection_rows outputs/xsum_10m_retrieval_ppl_v1/rows.jsonl \
    --output_dir "${output_root}" \
    --rank "${rank}" \
    --world_size 8 \
    --device cuda:0 \
    > "${output_root}/rank${rank}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
