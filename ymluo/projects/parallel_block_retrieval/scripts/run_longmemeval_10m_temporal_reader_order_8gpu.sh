#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

selection_root="outputs/longmemeval_10m_temporal_multivalue_state_ku_v1"

for order in chronological latest_first; do
  output_root="outputs/longmemeval_10m_temporal_reader_ku_dates_${order}_v1"
  mkdir -p "${output_root}"
  pids=()
  for partition in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES="${partition}" \
      /home/fdong/miniconda3/envs/moe/bin/python \
      src/evaluate_longmemeval_10m_selected_reader.py \
      --data_dir "data/longmemeval_10m_partition${partition}_v1" \
      --selection_rows "${selection_root}/part${partition}/rows.jsonl" \
      --output_dir "${output_root}/part${partition}" \
      --device cuda:0 \
      --question_types knowledge-update \
      --include_page_dates \
      --page_order "${order}" \
      > "${output_root}/part${partition}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
  fi
done
