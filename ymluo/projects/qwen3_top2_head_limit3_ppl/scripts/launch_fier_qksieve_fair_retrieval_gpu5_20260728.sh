#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RUN_ROOT=$ROOT/results/20260728_fier_qksieve_fair_retrieval_32k
TRACE_ROOT=$ROOT/results/20260728_qksieve_all_layer_bits_qwen3_32k/traces
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=5 "$PYTHON" -u \
  src/analyze_fier_qksieve_retrieval_fair_20260728.py \
  --trace "sports=$TRACE_ROOT/qwen3_4b_sports32k_all_layers.pt" \
  --trace "medicine=$TRACE_ROOT/qwen3_4b_medicine32k_all_layers.pt" \
  --output_dir "$RUN_ROOT" \
  --device cuda \
  --calibration_steps 8 \
  --query_shrinkage 0.75 \
  --key_stride 32 \
  --fier_group_size 32 \
  --budgets 0.01,0.02,0.04 \
  >"$LOG_ROOT/main.log" 2>&1
