#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/tree_retrieval_energy}"

python "${PROJECT_DIR}/src/analyze_qwen3_kv_tree_retrieval_energy.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-5000}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_max_tokens "${REQUIRE_MAX_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --layers "${LAYERS:-all}" \
  --kv_heads "${KV_HEADS:-all}" \
  --boundary_fraction "${BOUNDARY_FRACTION:-0.01}" \
  --leaf_fraction "${LEAF_FRACTION:-0.001}" \
  --leaf_size "${LEAF_SIZE:-0}" \
  --tree_fanout "${TREE_FANOUT:-10}" \
  --tree_branch_counts "${TREE_BRANCH_COUNTS:-5,5,5}" \
  --candidate_granularity "${CANDIDATE_GRANULARITY:-attention_head}" \
  --compute_oracle_baseline "${COMPUTE_ORACLE_BASELINE:-true}" \
  --save_token_rows "${SAVE_TOKEN_ROWS:-true}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_smoothing_window "${PLOT_SMOOTHING_WINDOW:-500}" \
  --plot_dpi "${PLOT_DPI:-180}"
