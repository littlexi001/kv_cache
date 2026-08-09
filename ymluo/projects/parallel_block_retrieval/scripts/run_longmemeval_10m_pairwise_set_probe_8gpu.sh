#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/parallel_block_retrieval

output_root="outputs/longmemeval_10m_pairwise_set_probe_all500_v1"
mkdir -p "${output_root}"

pids=()
for partition in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${partition}" \
    /home/fdong/miniconda3/envs/moe/bin/python \
    src/evaluate_longmemeval_10m_pairwise_set_utility_probe.py \
    --data_dir "data/longmemeval_10m_partition${partition}_v1" \
    --selection_rows "outputs/longmemeval_10m_evidence_state_all500_v1/part${partition}/rows.jsonl" \
    --state_rows "outputs/longmemeval_10m_evidence_state_all500_v1/part${partition}/states.jsonl" \
    --output_dir "${output_root}/part${partition}" \
    --device cuda:0 \
    > "${output_root}/part${partition}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
