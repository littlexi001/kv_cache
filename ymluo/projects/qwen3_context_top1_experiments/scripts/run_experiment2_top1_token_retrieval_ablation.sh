#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
DATA_PATH="${DATA_PATH:-${REPO_DIR}/ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/experiment2_top1_token_retrieval_ablation}"

python "${PROJECT_DIR}/src/experiment2_top1_token_retrieval_ablation.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_samples "${MAX_SAMPLES:-16}" \
  --seed "${SEED:-7}" \
  --top_ratio "${TOP_RATIO:-0.01}" \
  --query_last_tokens "${QUERY_LAST_TOKENS:-16}" \
  --ablation_layer "${ABLATION_LAYER:-last}" \
  --max_context_chars "${MAX_CONTEXT_CHARS:-24000}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --trust_remote_code "${TRUST_REMOTE_CODE:-true}" \
  --include_special_tokens "${INCLUDE_SPECIAL_TOKENS:-false}" \
  --random_keep_trials "${RANDOM_KEEP_TRIALS:-3}"
