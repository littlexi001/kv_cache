#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/traces" "$OUTPUT/logs"
cd "$ROOT"

run_topic() {
  local gpu="$1"
  local topic="$2"
  local seed="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$MODEL" \
    --output_path "$OUTPUT/traces/${topic}.pt" \
    --topic "$topic" \
    --history_tokens 96000 \
    --steps 8 \
    --layers 0,8,17,26,35 \
    --prefill_query_tail_tokens 8 \
    --prefill_chunk_tokens 2048 \
    --seed "$seed" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"$OUTPUT/logs/${topic}.log" 2>&1
  touch "$OUTPUT/${topic}_COMPLETE"
}

run_topic "1,2" sports 20260851 &
pid1=$!
run_topic "3,4" medicine 20260852 &
pid2=$!
wait "$pid1"
wait "$pid2"
touch "$OUTPUT/ALL_COMPLETE"
