#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

selection_root="outputs/longmemeval_10m_evidence_state_all500_v1"
IFS=',' read -r -a gpu_ids <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"

for config in dates_retrieval chronological_nodates dates_latest_first; do
  output_root="outputs/longmemeval_10m_all500_${config}_reader_v1"
  mkdir -p "${output_root}"
  for ((batch_start = 0; batch_start < 8; batch_start += ${#gpu_ids[@]})); do
    pids=()
    for slot in "${!gpu_ids[@]}"; do
      partition=$((batch_start + slot))
      if ((partition >= 8)); then
        continue
      fi
      if [[ -s "${output_root}/part${partition}/summary.json" ]]; then
        continue
      fi
      extra_args=()
      if [[ "${config}" == "dates_retrieval" ]]; then
        extra_args+=(--include_page_dates --page_order retrieval)
      elif [[ "${config}" == "chronological_nodates" ]]; then
        extra_args+=(--page_order chronological)
      else
        extra_args+=(--include_page_dates --page_order latest_first)
      fi
      CUDA_VISIBLE_DEVICES="${gpu_ids[slot]}" \
        /home/fdong/miniconda3/envs/moe/bin/python \
        src/evaluate_longmemeval_10m_selected_reader.py \
        --data_dir "data/longmemeval_10m_partition${partition}_v1" \
        --selection_rows "${selection_root}/part${partition}/rows.jsonl" \
        --output_dir "${output_root}/part${partition}" \
        --device cuda:0 \
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
    if ((status != 0)); then
      exit "${status}"
    fi
  done
done
