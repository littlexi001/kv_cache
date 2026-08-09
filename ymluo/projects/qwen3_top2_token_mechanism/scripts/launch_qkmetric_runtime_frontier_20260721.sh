#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism
OUTPUT_ROOT=${1:-${ROOT}/outputs/qkmetric_runtime_frontier_128k_20260721}
LOG_ROOT=${ROOT}/artifacts/20260721_numeric_pruning_frontier/qkmetric_runtime_frontier_logs
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

topics=(medicine politics computer space)
gpu_pairs=(0,1 2,3 4,5 6,7)
pids=()

for index in "${!topics[@]}"; do
  topic=${topics[$index]}
  gpu_pair=${gpu_pairs[$index]}
  (
    bash "${ROOT}/scripts/run_qkmetric_128k_20260721.sh" \
      "${gpu_pair}" "${topic}" 1024 "${OUTPUT_ROOT}" 48 0.06
    bash "${ROOT}/scripts/run_qkmetric_128k_20260721.sh" \
      "${gpu_pair}" "${topic}" 1024 "${OUTPUT_ROOT}" 64 0.04
  ) > "${LOG_ROOT}/${topic}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
