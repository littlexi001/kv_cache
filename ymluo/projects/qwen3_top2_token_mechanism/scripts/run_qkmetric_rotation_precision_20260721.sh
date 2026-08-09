#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism
ARTIFACTS=${ROOT}/artifacts/20260721_numeric_pruning_frontier
PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
export PYTHONPATH=${ROOT}/src:${PYTHONPATH:-}

TRACE_ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260717_delta_qkv_traces_32k_s16
SPORTS=${TRACE_ROOT}/sports.pt
MEDICINE=${TRACE_ROOT}/medicine.pt
QWEN128=${ROOT}/artifacts/20260720_oneshot_combinations/medicine_128k_layer16_s16.pt

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
  "${ROOT}/src/analyze_qk_metric_rotation_precision.py" \
  --trace_paths "${SPORTS}" "${MEDICINE}" \
  --output_path "${ARTIFACTS}/qkmetric_rotation_precision_llama.json" \
  --layers 0,8,16,24,31 \
  --ranks 32,48,64 \
  --train_steps 4 \
  --test_start_step 8 \
  --test_steps 8 \
  > "${ARTIFACTS}/qkmetric_rotation_precision_llama.log" 2>&1 &
llama_pid=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" \
  "${ROOT}/src/analyze_qk_metric_rotation_precision.py" \
  --trace_paths "${QWEN128}" \
  --output_path "${ARTIFACTS}/qkmetric_rotation_precision_qwen128.json" \
  --layers 16 \
  --ranks 32,48,64 \
  --train_steps 4 \
  --test_start_step 8 \
  --test_steps 8 \
  > "${ARTIFACTS}/qkmetric_rotation_precision_qwen128.log" 2>&1 &
qwen_pid=$!

status=0
wait "${llama_pid}" || status=1
wait "${qwen_pid}" || status=1
exit "${status}"
