#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_value_decomposition}"

python "${PROJECT_DIR}/src/analyze_qwen3_attention_value_decomposition.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-5000}" \
  --eval_tokens "${EVAL_TOKENS:-5000}" \
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
  --split_mode "${SPLIT_MODE:-mass}" \
  --top_values "${TOP_VALUES:-0.9}" \
  --tail_values "${TAIL_VALUES:-0.1}" \
  --compute_vector_stats "${COMPUTE_VECTOR_STATS:-true}" \
  --compute_ppl "${COMPUTE_PPL:-true}" \
  --ppl_modes "${PPL_MODES:-full,top0p9,tail0p1}" \
  --ppl_renormalize_selected "${PPL_RENORMALIZE_SELECTED:-false}"
