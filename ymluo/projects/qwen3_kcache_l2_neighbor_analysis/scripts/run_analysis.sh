#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/k_l2_neighbors}"

python "${PROJECT_DIR}/src/analyze_qwen3_kcache_l2_neighbors.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-5000}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_max_tokens "${REQUIRE_MAX_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --rope_max_position_embeddings "${ROPE_MAX_POSITION_EMBEDDINGS:-8192}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --neighbor_count "${NEIGHBOR_COUNT:-5}" \
  --neighbor_scope "${NEIGHBOR_SCOPE:-all}" \
  --variants "${VARIANTS:-raw}" \
  --distance_device "${DISTANCE_DEVICE:-auto}" \
  --save_neighbor_csv "${SAVE_NEIGHBOR_CSV:-true}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --make_heatmaps "${MAKE_HEATMAPS:-true}" \
  --heatmap_max_tokens "${HEATMAP_MAX_TOKENS:-1500}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --point_alpha "${POINT_ALPHA:-0.35}"
