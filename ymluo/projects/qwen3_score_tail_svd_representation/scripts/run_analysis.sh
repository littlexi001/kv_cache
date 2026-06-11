#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/score_tail_svd_representation}"

python "${PROJECT_DIR}/src/analyze_qwen3_score_tail_svd_representation.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-5000}" \
  --eval_tokens "${EVAL_TOKENS:-1024}" \
  --chunk_size "${CHUNK_SIZE:-128}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_total_tokens "${REQUIRE_TOTAL_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --svd_components "${SVD_COMPONENTS:-8}" \
  --query_stride "${QUERY_STRIDE:-8}" \
  --max_query_rows_per_layer_head "${MAX_QUERY_ROWS_PER_LAYER_HEAD:-512}" \
  --max_vectors_per_group "${MAX_VECTORS_PER_GROUP:-4096}" \
  --max_score_values_per_group "${MAX_SCORE_VALUES_PER_GROUP:-200000}" \
  --representations "${REPRESENTATIONS:-key,value,weighted_value}"
