#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/qabs_attention_only_benchmark_${STAMP}}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -u src/benchmark_qabs_attention_only.py \
  --output_dir "${OUTPUT_DIR}" \
  --histories "${HISTORIES:-1024,4096,8192,16384,32768}" \
  --mode "${MODE:-qabs8cand3reusefinal}" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --batch_count "${BATCH_COUNT:-1}" \
  --head_count "${HEAD_COUNT:-16}" \
  --head_dim "${HEAD_DIM:-128}" \
  --layer_count "${LAYER_COUNT:-28}" \
  --qabs_dim_count "${QABS_DIM_COUNT:-8}" \
  --qabs_candidate_fraction "${QABS_CANDIDATE_FRACTION:-0.03}" \
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --protect_sink_tokens "${PROTECT_SINK_TOKENS:-10}" \
  --protect_recent_tokens "${PROTECT_RECENT_TOKENS:-10}" \
  --always_keep_self "${ALWAYS_KEEP_SELF:-true}" \
  --partial_impl "${PARTIAL_IMPL:-cuda_dim_major}" \
  --use_cuda_kernels "${USE_CUDA_KERNELS:-true}" \
  --use_cuda_full_scores "${USE_CUDA_FULL_SCORES:-true}" \
  --use_cuda_final_attention "${USE_CUDA_FINAL_ATTENTION:-true}" \
  --warmup_iterations "${WARMUP_ITERATIONS:-10}" \
  --iterations "${ITERATIONS:-50}" \
  --profile_iterations "${PROFILE_ITERATIONS:-10}" \
  --seed "${SEED:-1234}"
