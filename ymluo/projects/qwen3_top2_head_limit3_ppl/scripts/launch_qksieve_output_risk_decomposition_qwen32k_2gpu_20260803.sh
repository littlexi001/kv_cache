#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_output_risk_decomposition_qwen32k_2topic_v2}"
GPU_SPORTS="${GPU_SPORTS:-0}"
GPU_MEDICINE="${GPU_MEDICINE:-1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"

run_topic() {
  local gpu="$1"
  local topic="$2"
  mkdir -p "$OUTPUT/$topic"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    "$ROOT/src/analyze_qksieve_output_risk_budget_20260803.py" \
    --trace "$TRACE_ROOT/$topic.pt" \
    --output_dir "$OUTPUT/$topic" \
    --model_name_or_path "$MODEL" \
    --device cuda \
    --fixed_top_k 1280 \
    --fixed_top_ks 640,1280,2560,4096 \
    --global_top_ks 1280,2560 \
    --coverage_targets 0.90,0.95,0.975,0.99 \
    --minimum_top_k 1 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    >"$OUTPUT/logs/$topic.log" 2>&1
  touch "$OUTPUT/${topic}_COMPLETE"
}

run_topic "$GPU_SPORTS" sports &
pid0=$!
run_topic "$GPU_MEDICINE" medicine &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$OUTPUT/ALL_COMPLETE"
