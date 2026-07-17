#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fdong/hrj/prove/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/head_evidence_attention_8k_20260714}"

"${PYTHON_BIN}" "${ROOT}/src/run_head_evidence_attention.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --model_label qwen3_0p6b \
  --output_dir "${OUTPUT_DIR}" \
  --lengths 8192 \
  --depths 50 \
  --seeds 0,1,2,3,4,5,6,7 \
  --distractor_counts 16 \
  --rule_gap_tokens 512 \
  --chain_lengths 2 \
  --competitor_counts 0,4 \
  --top_fraction 0.02 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation sdpa \
  --prefill_chunk_size 4096 \
  "$@"

"${PYTHON_BIN}" "${ROOT}/src/summarize_head_evidence_attention.py" \
  --input_dir "${OUTPUT_DIR}"
