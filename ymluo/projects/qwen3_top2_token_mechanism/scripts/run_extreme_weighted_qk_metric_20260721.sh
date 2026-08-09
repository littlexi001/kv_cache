#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism
ARTIFACTS=${ROOT}/artifacts/20260721_numeric_pruning_frontier
TRACE_ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260717_delta_qkv_traces_32k_s16
PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
export PYTHONPATH=${ROOT}/src:${PYTHONPATH:-}

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
  "${ROOT}/src/analyze_extreme_weighted_qk_metric.py" \
  --trace_paths "${TRACE_ROOT}/sports.pt" "${TRACE_ROOT}/medicine.pt" \
  --output_path "${ARTIFACTS}/extreme_weighted_qkmetric_llama.json" \
  --layers 0,8,16,24,31 \
  > "${ARTIFACTS}/extreme_weighted_qkmetric_llama.log" 2>&1 &
llama_pid=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" \
  "${ROOT}/src/analyze_extreme_weighted_qk_metric.py" \
  --trace_paths "${ROOT}/artifacts/20260720_oneshot_combinations/medicine_128k_layer16_s16.pt" \
  --output_path "${ARTIFACTS}/extreme_weighted_qkmetric_qwen128.json" \
  --layers 16 \
  > "${ARTIFACTS}/extreme_weighted_qkmetric_qwen128.log" 2>&1 &
qwen_pid=$!

status=0
wait "${llama_pid}" || status=1
wait "${qwen_pid}" || status=1
exit "${status}"
