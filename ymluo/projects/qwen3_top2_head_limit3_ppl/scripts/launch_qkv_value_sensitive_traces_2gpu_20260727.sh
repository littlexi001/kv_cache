#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-$ROOT/results/20260727_qkv_value_sensitive_32k}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT/traces" "$OUTPUT/logs"
cd "$ROOT"

run_trace() {
  local gpu="$1"
  local topic="$2"
  local output="$OUTPUT/traces/qwen3_4b_${topic}_qkv.pt"
  local log="$OUTPUT/logs/${topic}.log"
  if [[ -s "$output" ]]; then
    echo "SKIP ${topic}"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$MODEL" \
    --output_path "$output" \
    --topic "$topic" \
    --history_tokens 32000 \
    --steps 32 \
    --layers "0,8,17,26,35" \
    --prefill_chunk_tokens 2048 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$log" 2>&1
}

run_trace 0 sports &
pid0=$!
run_trace 1 medicine &
pid1=$!
wait "$pid0"
wait "$pid1"
touch "$OUTPUT/TRACES_COMPLETE"
