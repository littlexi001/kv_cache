#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPU="${GPU:-1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_lowbit_layout_benchmark}"
LENGTHS="${LENGTHS:-8192,16384,32768,65536,131072}"
PROFILES="${PROFILES:-auto240_reference,fixed420_b128,fixed410_b112,fixed400_b80}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-100}"

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -u src/benchmark_qksieve_lowbit_layouts_20260731.py \
  --lengths "${LENGTHS}" \
  --profiles "${PROFILES}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/run.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
