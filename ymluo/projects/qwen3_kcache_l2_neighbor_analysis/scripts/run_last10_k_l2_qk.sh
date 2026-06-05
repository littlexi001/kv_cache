#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-${PROJECT_DIR}/data/needle_in_haystack/prompts/niah_len8000_depth50.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/last10_k_l2_qk}"

python "${PROJECT_DIR}/src/analyze_qwen3_last10_k_l2_qk.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-8192}" \
  --truncate_side "${TRUNCATE_SIDE:-tail}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_max_tokens "${REQUIRE_MAX_TOKENS:-false}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --rope_max_position_embeddings "${ROPE_MAX_POSITION_EMBEDDINGS:-8192}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --last_token_count "${LAST_TOKEN_COUNT:-10}" \
  --needle_text "${NEEDLE_TEXT:-The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.}" \
  --qk_reduce "${QK_REDUCE:-mean}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --line_alpha "${LINE_ALPHA:-0.85}" \
  --line_width "${LINE_WIDTH:-0.9}" \
  --sink_token_count "${SINK_TOKEN_COUNT:-16}" \
  --zoom_first_tokens "${ZOOM_FIRST_TOKENS:-128}" \
  --make_attention_weight_plots "${MAKE_ATTENTION_WEIGHT_PLOTS:-true}" \
  --make_zoom_plots "${MAKE_ZOOM_PLOTS:-true}" \
  --make_per_query_head_plots "${MAKE_PER_QUERY_HEAD_PLOTS:-true}" \
  --make_sink_heatmaps "${MAKE_SINK_HEATMAPS:-true}" \
  --save_plot_data "${SAVE_PLOT_DATA:-true}" \
  --save_csv "${SAVE_CSV:-false}"
