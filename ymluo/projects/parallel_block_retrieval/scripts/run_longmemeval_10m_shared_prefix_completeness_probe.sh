#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

output_root="outputs/longmemeval_10m_shared_prefix_completeness_probe_all500_v1"
mkdir -p "${output_root}"

status=0
IFS=',' read -r -a gpus <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
for ((base = 0; base < 8; base += ${#gpus[@]})); do
  pids=()
  for ((slot = 0; slot < ${#gpus[@]}; slot++)); do
    partition=$((base + slot))
    if ((partition >= 8)); then
      break
    fi
    CUDA_VISIBLE_DEVICES="${gpus[slot]}" \
      /home/fdong/miniconda3/envs/moe/bin/python \
      src/evaluate_longmemeval_10m_shared_prefix_completeness_probe.py \
      --data_dir "data/longmemeval_10m_partition${partition}_v1" \
      --selection_rows "outputs/longmemeval_10m_evidence_state_all500_v1/part${partition}/rows.jsonl" \
      --state_rows "outputs/longmemeval_10m_evidence_state_all500_v1/part${partition}/states.jsonl" \
      --output_dir "${output_root}/part${partition}" \
      --device cuda:0 \
      > "${output_root}/part${partition}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
done
exit "${status}"
