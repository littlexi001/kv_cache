#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/experiment1_full_context_not_optimal}"

python "${PROJECT_DIR}/src/experiment1_full_context_not_optimal.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples "${NUM_SAMPLES:-48}" \
  --num_irrelevant "${NUM_IRRELEVANT:-8}" \
  --num_semantic_distractors "${NUM_SEMANTIC_DISTRACTORS:-4}" \
  --seed "${SEED:-13}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-12}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --trust_remote_code "${TRUST_REMOTE_CODE:-true}" \
  --do_generate "${DO_GENERATE:-true}"
