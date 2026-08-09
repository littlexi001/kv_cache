#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TRACE_ROOT="$ROOT/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces"
OUTPUT="${OUTPUT:-$ROOT/results/20260803_residual_priority_qwen32k_2topic_v1}"
CANDIDATES="proxy,residualfp_proxy,residual8_proxy,residual4_proxy,residualfp_exact"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/logs"
cd "$ROOT"

run_topic() {
  local gpu="$1"
  local topic="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qksieve_prefill_tail_calibration_20260803.py \
    --trace "$TRACE_ROOT/${topic}.pt" \
    --output_dir "$OUTPUT/$topic" \
    --model_name_or_path "$MODEL" \
    --top_k 1280 \
    --value_rank 16 \
    --value_bits 4 \
    --value_metrics raw,wo_group \
    --candidate_modes "$CANDIDATES" \
    --calibration_counts 8 \
    >"$OUTPUT/logs/${topic}.log" 2>&1
  touch "$OUTPUT/${topic}_COMPLETE"
}

run_topic 0 sports &
pid0=$!
run_topic 1 medicine &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$OUTPUT/ALL_COMPLETE"
