#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TRACE="${TRACE:-${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_length_order_statistics_computer128k_gpu1_v1}"
GPU="${GPU:-1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
  "$ROOT/src/analyze_qksieve_length_order_statistics_20260803.py" \
  --trace "$TRACE" \
  --output_dir "$OUTPUT" \
  --device cuda \
  --lengths 4096,8192,16384,32768,65536,98304,131008 \
  --top_k 1280 \
  --key_sample_stride 32 \
  --query_shrinkage 0.75 \
  --key_rate_budget 15 \
  --calibration_samples 256 \
  >"$OUTPUT/run.log" 2>&1

touch "$OUTPUT/ALL_COMPLETE"
