#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_article_01.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
ANALYSIS_LEVEL="${ANALYSIS_LEVEL:-token}"
GRAPH_MODE="${GRAPH_MODE:-topk}"
TOP_K="${TOP_K:-5}"
RADIUS_THRESHOLD="${RADIUS_THRESHOLD:-}"
MAX_RADIUS_NEIGHBORS="${MAX_RADIUS_NEIGHBORS:-200}"
SIMILARITY="${SIMILARITY:-cos}"
HIST_BINS="${HIST_BINS:-auto}"
SAVE_NEIGHBORS="${SAVE_NEIGHBORS:-0}"
CENTER_TOKENS="${CENTER_TOKENS:-1}"
KEY_TRANSFORM="${KEY_TRANSFORM:-}"
PC_REMOVE_COUNT="${PC_REMOVE_COUNT:-0}"
ALLOW_LONGER_THAN_MODEL_MAX="${ALLOW_LONGER_THAN_MODEL_MAX:-0}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-transformers451/bin/python" ]]; then
    PYTHON=".venv-transformers451/bin/python"
  else
    PYTHON="python3"
  fi
fi

args=(
  fdong_seq_compress/src/run_k_similarity_graph_probe.py
  --model-path "$MODEL_PATH"
  --text-path "$TEXT_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-tokens "$MAX_TOKENS"
  --layers "$LAYERS"
  --heads "$HEADS"
  --analysis-level "$ANALYSIS_LEVEL"
  --graph-mode "$GRAPH_MODE"
  --top-k "$TOP_K"
  --max-radius-neighbors "$MAX_RADIUS_NEIGHBORS"
  --similarity "$SIMILARITY"
  --pc-remove-count "$PC_REMOVE_COUNT"
  "--hist-bins=$HIST_BINS"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$SAVE_NEIGHBORS" == "1" ]]; then
  args+=(--save-neighbors)
fi

if [[ -n "$RADIUS_THRESHOLD" ]]; then
  args+=(--radius-threshold "$RADIUS_THRESHOLD")
fi

if [[ -n "$KEY_TRANSFORM" ]]; then
  args+=(--key-transform "$KEY_TRANSFORM")
fi

if [[ "$CENTER_TOKENS" == "1" ]]; then
  args+=(--center-tokens)
fi

if [[ "$ALLOW_LONGER_THAN_MODEL_MAX" == "1" ]]; then
  args+=(--allow-longer-than-model-max)
fi

"$PYTHON" "${args[@]}"
