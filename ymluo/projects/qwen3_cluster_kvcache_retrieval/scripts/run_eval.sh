#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/qwen3_cluster_kvcache_retrieval}"

python "${PROJECT_DIR}/src/run_cluster_kvcache_eval.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --modes "${MODES:-baseline,cluster}" \
  --prefill_tokens "${PREFILL_TOKENS:-100000}" \
  --eval_tokens "${EVAL_TOKENS:-512}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-512}" \
  --cluster_size "${CLUSTER_SIZE:-50}" \
  --keep_ratio "${KEEP_RATIO:-0.02}" \
  --edge_ratio "${EDGE_RATIO:-0.01}" \
  --force_endpoints "${FORCE_ENDPOINTS:-true}" \
  --endpoints_count_in_budget "${ENDPOINTS_COUNT_IN_BUDGET:-true}" \
  --max_chars "${MAX_CHARS:-160000000}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --profile_attention "${PROFILE_ATTENTION:-true}" \
  --profile_modules "${PROFILE_MODULES:-false}" \
  --warmup_eval_tokens "${WARMUP_EVAL_TOKENS:-8}" \
  --save_token_timings "${SAVE_TOKEN_TIMINGS:-true}"
