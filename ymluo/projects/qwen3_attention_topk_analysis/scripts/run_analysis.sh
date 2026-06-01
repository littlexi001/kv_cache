#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_topk}"

python "${PROJECT_DIR}/src/analyze_qwen3_attention_topk.py" \
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
  --heads "${HEADS:-all}" \
  --top_k "${TOP_K:-5}" \
  --include_self "${INCLUDE_SELF:-true}" \
  --save_token_rows "${SAVE_TOKEN_ROWS:-true}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --point_alpha "${POINT_ALPHA:-0.35}"
