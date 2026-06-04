#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_textbook_distributed_systems.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-2000}"
DECODE_START="${DECODE_START:-1000}"
QUERY_STRIDE="${QUERY_STRIDE:-8}"
MAX_QUERIES="${MAX_QUERIES:-128}"
LAYERS="${LAYERS:-27}"
Q_HEADS="${Q_HEADS:-0}"
SIMILARITY="${SIMILARITY:-l2}"
GRAPH_TOP_K="${GRAPH_TOP_K:-10}"
GRAPH_HOPS="${GRAPH_HOPS:-1}"
GRAPH_DIRECTION="${GRAPH_DIRECTION:-both}"
LOCAL_WINDOW="${LOCAL_WINDOW:-128}"
SEED_COUNT="${SEED_COUNT:-8}"
MAX_CANDIDATES="${MAX_CANDIDATES:-256}"
METHODS="${METHODS:-local,random_local_size,random_max_candidates,local_topq_graph,local_graph_all}"
TOP_ATTENTION_KS="${TOP_ATTENTION_KS:-1,5,10}"
ALWAYS_INCLUDE_POSITIONS="${ALWAYS_INCLUDE_POSITIONS:-0:10}"
SINK_POSITIONS="${SINK_POSITIONS:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"
SAVE_EXAMPLES="${SAVE_EXAMPLES:-20}"
ALLOW_LONGER_THAN_MODEL_MAX="${ALLOW_LONGER_THAN_MODEL_MAX:-0}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-transformers451/bin/python" ]]; then
    PYTHON=".venv-transformers451/bin/python"
  else
    PYTHON="python3"
  fi
fi

args=(
  fdong_seq_compress/src/run_k_graph_attention_recall.py
  --model-path "$MODEL_PATH"
  --text-path "$TEXT_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max-tokens "$MAX_TOKENS"
  --decode-start "$DECODE_START"
  --query-stride "$QUERY_STRIDE"
  --max-queries "$MAX_QUERIES"
  --layers "$LAYERS"
  --q-heads "$Q_HEADS"
  --similarity "$SIMILARITY"
  --graph-top-k "$GRAPH_TOP_K"
  --graph-hops "$GRAPH_HOPS"
  --graph-direction "$GRAPH_DIRECTION"
  --local-window "$LOCAL_WINDOW"
  --seed-count "$SEED_COUNT"
  --max-candidates "$MAX_CANDIDATES"
  --methods "$METHODS"
  --top-attention-ks "$TOP_ATTENTION_KS"
  --always-include-positions "$ALWAYS_INCLUDE_POSITIONS"
  --sink-positions "$SINK_POSITIONS"
  --random-seed "$RANDOM_SEED"
  --save-examples "$SAVE_EXAMPLES"
)

if [[ -n "$OUTPUT_DIR" ]]; then
  args+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$ALLOW_LONGER_THAN_MODEL_MAX" == "1" ]]; then
  args+=(--allow-longer-than-model-max)
fi

"$PYTHON" "${args[@]}"
