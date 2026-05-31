#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
LAYERS="${LAYERS:-all}"
Q_HEADS="${Q_HEADS:-all}"
QUERY_STRIDE="${QUERY_STRIDE:-8}"
MIN_QUERY_INDEX="${MIN_QUERY_INDEX:-2}"
TOP_K="${TOP_K:-10}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-transformers451/bin/python" ]]; then
    PYTHON=".venv-transformers451/bin/python"
  else
    PYTHON="python3"
  fi
fi

args=(
  fdong_seq_compress/src/run_qk_common_direction_probe.py
  --model-path "$MODEL_PATH"
  --text-path "$TEXT_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-tokens "$MAX_TOKENS"
  --layers "$LAYERS"
  --q-heads "$Q_HEADS"
  --query-stride "$QUERY_STRIDE"
  --min-query-index "$MIN_QUERY_INDEX"
  --top-k "$TOP_K"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

"$PYTHON" "${args[@]}"
