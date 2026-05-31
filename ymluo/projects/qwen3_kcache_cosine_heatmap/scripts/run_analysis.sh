#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kcache_cosine_heatmap}"

python "${PROJECT_DIR}/src/analyze_qwen3_kcache_cosine_heatmap.py" \
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
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --similarity_device "${SIMILARITY_DEVICE:-auto}" \
  --similarity_dtype "${SIMILARITY_DTYPE:-float32}" \
  --summary_percentiles "${SUMMARY_PERCENTILES:-1,5,25,50,75,95,99}" \
  --summary_sample_size "${SUMMARY_SAMPLE_SIZE:-1000000}" \
  --sample_seed "${SAMPLE_SEED:-1234}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_max_tokens "${PLOT_MAX_TOKENS:-5000}" \
  --figure_size "${FIGURE_SIZE:-7.5}" \
  --plot_dpi "${PLOT_DPI:-180}" \
  --cmap "${CMAP:-coolwarm}" \
  --vmin "${VMIN:--1.0}" \
  --vmax "${VMAX:-1.0}" \
  --save_similarity_tensors "${SAVE_SIMILARITY_TENSORS:-false}" \
  --saved_matrix_dtype "${SAVED_MATRIX_DTYPE:-float16}" \
  --write_token_csv "${WRITE_TOKEN_CSV:-true}" \
  --histogram_bins "${HISTOGRAM_BINS:-200}" \
  --histogram_min "${HISTOGRAM_MIN:--1.0}" \
  --histogram_max "${HISTOGRAM_MAX:-1.0}" \
  --compute_pairwise_distances "${COMPUTE_PAIRWISE_DISTANCES:-true}" \
  --distance_cache_types "${DISTANCE_CACHE_TYPES:-k,v}" \
  --distance_bins "${DISTANCE_BINS:-200}" \
  --distance_min "${DISTANCE_MIN:-0.0}" \
  --distance_max "${DISTANCE_MAX:-0.0}" \
  --compute_top_p_previous_distances "${COMPUTE_TOP_P_PREVIOUS_DISTANCES:-true}" \
  --top_p_previous_cache_types "${TOP_P_PREVIOUS_CACHE_TYPES:-k}" \
  --top_p_previous_count "${TOP_P_PREVIOUS_COUNT:-5}" \
  --save_top_p_previous_token_rows "${SAVE_TOP_P_PREVIOUS_TOKEN_ROWS:-true}" \
  --compute_cache_clusters "${COMPUTE_CACHE_CLUSTERS:-false}" \
  --cluster_cache_types "${CLUSTER_CACHE_TYPES:-k,v}" \
  --cluster_count "${CLUSTER_COUNT:-32}" \
  --cluster_iterations "${CLUSTER_ITERATIONS:-30}" \
  --cluster_seed "${CLUSTER_SEED:-1234}" \
  --cluster_normalize "${CLUSTER_NORMALIZE:-false}" \
  --save_cluster_assignments "${SAVE_CLUSTER_ASSIGNMENTS:-false}"
