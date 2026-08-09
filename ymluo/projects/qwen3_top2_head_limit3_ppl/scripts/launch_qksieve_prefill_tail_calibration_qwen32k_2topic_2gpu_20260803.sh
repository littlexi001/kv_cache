#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/traces" "$OUTPUT/analysis" "$OUTPUT/logs"
cd "$ROOT"

run_topic() {
  local gpu="$1"
  local topic="$2"
  local seed="$3"
  local trace="$OUTPUT/traces/${topic}.pt"
  local analysis="$OUTPUT/analysis/${topic}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$MODEL" \
    --output_path "$trace" \
    --topic "$topic" \
    --history_tokens 32000 \
    --steps 32 \
    --layers 0,8,17,26,35 \
    --prefill_query_tail_tokens 8 \
    --prefill_chunk_tokens 2048 \
    --seed "$seed" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$OUTPUT/logs/${topic}_capture.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/analyze_qksieve_prefill_tail_calibration_20260803.py \
    --trace "$trace" \
    --output_dir "$analysis" \
    --model_name_or_path "$MODEL" \
    --top_k 1280 \
    --value_rank 16 \
    --value_bits 4 \
    --calibration_counts 1,2,4,8 \
    >"$OUTPUT/logs/${topic}_analysis.log" 2>&1
  touch "$OUTPUT/${topic}_COMPLETE"
}

run_topic 0 sports 20260841 &
pid0=$!
run_topic 1 medicine 20260842 &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$OUTPUT/ALL_COMPLETE"
