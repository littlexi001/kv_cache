#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_llama4k_religion_qkv_trace_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/traces" "$OUTPUT/logs"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=5 "$PYTHON" -u \
  src/collect_real_qk_trace_20260715.py \
  --model_name_or_path "$MODEL" \
  --output_path "$OUTPUT/traces/religion.pt" \
  --topic religion \
  --history_tokens 3968 \
  --steps 32 \
  --layers 0,8,16,24,31 \
  --prefill_query_tail_tokens 8 \
  --prefill_chunk_tokens 1024 \
  --seed 20260835 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$OUTPUT/logs/capture.log" 2>&1

touch "$OUTPUT/ALL_COMPLETE"
