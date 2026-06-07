#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kkl2_graph_ppl}"

python "${PROJECT_DIR}/src/evaluate_qwen3_kkl2_graph_ppl.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-2048}" \
  --eval_tokens "${EVAL_TOKENS:-2048}" \
  --eval_last_tokens_only "${EVAL_LAST_TOKENS_ONLY:-false}" \
  --chunk_size "${CHUNK_SIZE:-128}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_total_tokens "${REQUIRE_TOTAL_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --compute_graph_ppl "${COMPUTE_GRAPH_PPL:-true}" \
  --layers "${LAYERS:-all}" \
  --kv_heads "${KV_HEADS:-all}" \
  --boundary_fraction "${BOUNDARY_FRACTION:-0.005}" \
  --middle_fraction "${MIDDLE_FRACTION:-0.01}" \
  --seed_count "${SEED_COUNT:-16}" \
  --graph_degree "${GRAPH_DEGREE:-20}" \
  --graph_update_interval "${GRAPH_UPDATE_INTERVAL:-100}" \
  --graph_update_mode "${GRAPH_UPDATE_MODE:-block_previous}" \
  --max_hops "${MAX_HOPS:-2}" \
  --candidate_granularity "${CANDIDATE_GRANULARITY:-kv_head_union}" \
  --compute_overlap_metrics "${COMPUTE_OVERLAP_METRICS:-true}" \
  --position_bins "${POSITION_BINS:-20}"
