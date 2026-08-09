#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

selection_root="outputs/longmemeval_10m_temporal_multivalue_state_ku_v1"
IFS=',' read -r -a gpu_ids <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"

for config in dates_retrieval chronological_nodates; do
  output_root="outputs/longmemeval_10m_temporal_reader_ku_${config}_v1"
  mkdir -p "${output_root}"
  for batch_start in $(seq 0 "${#gpu_ids[@]}" 7); do
    pids=()
    for slot in "${!gpu_ids[@]}"; do
      partition=$((batch_start + slot))
      if [[ "${partition}" -ge 8 ]]; then
        continue
      fi
      if [[ -s "${output_root}/part${partition}/summary.json" ]]; then
        continue
      fi
      extra_args=()
      if [[ "${config}" == "dates_retrieval" ]]; then
        extra_args+=(--include_page_dates --page_order retrieval)
      else
        extra_args+=(--page_order chronological)
      fi
      CUDA_VISIBLE_DEVICES="${gpu_ids[slot]}" \
        /home/fdong/miniconda3/envs/moe/bin/python \
        src/evaluate_longmemeval_10m_selected_reader.py \
        --data_dir "data/longmemeval_10m_partition${partition}_v1" \
        --selection_rows "${selection_root}/part${partition}/rows.jsonl" \
        --output_dir "${output_root}/part${partition}" \
        --device cuda:0 \
        --question_types knowledge-update \
        "${extra_args[@]}" \
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
done
