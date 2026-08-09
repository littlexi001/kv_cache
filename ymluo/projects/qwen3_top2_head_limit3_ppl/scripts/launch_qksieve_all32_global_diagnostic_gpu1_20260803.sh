#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE="${TRACE:-${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_llama4k_religion_all32_global_diagnostic_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -u \
  src/analyze_qksieve_output_risk_budget_20260803.py \
  --trace "$TRACE" \
  --output_dir "$OUTPUT" \
  --model_name_or_path "$MODEL" \
  --device cuda \
  --fixed_top_k 1280 \
  --global_top_ks 1280 \
  --global_priority_names calibrated_grouprisk4,jointrmse4,jointoracle4 \
  --coverage_targets 0.90 \
  --minimum_top_k 1 \
  --maximum_top_k 0 \
  --key_rate_budget 15 \
  --value_rank 16 \
  --value_bits 4 \
  --risk_bits 4 \
  --score_calibration_samples 256 \
  >"$OUTPUT/logs/analyze.log" 2>&1

touch "$OUTPUT/ALL_COMPLETE"
