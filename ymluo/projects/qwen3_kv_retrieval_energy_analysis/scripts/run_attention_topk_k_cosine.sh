#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_topk_k_cosine}"

python "${PROJECT_DIR}/src/analyze_attention_topk_k_cosine.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_tokens "${MAX_TOKENS:-4096}" \
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
  --top_fraction "${TOP_FRACTION:-0.02}" \
  --query_stride "${QUERY_STRIDE:-128}" \
  --min_visible_tokens "${MIN_VISIBLE_TOKENS:-128}" \
  --random_samples_per_query "${RANDOM_SAMPLES_PER_QUERY:-1}" \
  --sample_seed "${SAMPLE_SEED:-0}" \
  --percentiles "${PERCENTILES:-0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99}" \
  --make_heatmaps "${MAKE_HEATMAPS:-true}" \
  --heatmap_query_positions "${HEATMAP_QUERY_POSITIONS:-0.25,0.5,0.75,last}" \
  --heatmap_layers "${HEATMAP_LAYERS:-0,13,27}" \
  --heatmap_kv_heads "${HEATMAP_KV_HEADS:-0}" \
  --heatmap_attention_heads "${HEATMAP_ATTENTION_HEADS:-auto}" \
  --heatmap_max_vectors "${HEATMAP_MAX_VECTORS:-192}" \
  --plot_dpi "${PLOT_DPI:-180}"
