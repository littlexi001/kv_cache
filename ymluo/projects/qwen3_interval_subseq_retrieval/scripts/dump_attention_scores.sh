#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/dump_attention_scores.py" \
  --run_dir "${RUN_DIR:-${PROJECT_DIR}/outputs/train/${RUN_NAME:-unet8-interval1-lm-ddp}}" \
  --ckpt_step "${CKPT_STEP:-2000}" \
  --config_dir "${CONFIG_DIR:-/mnt/workspace/Qwen3-0.6B}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --query_positions "${QUERY_POSITIONS:-all}" \
  --save_raw_scores "${SAVE_RAW_SCORES:-true}" \
  --save_probabilities "${SAVE_PROBABILITIES:-true}" \
  --save_format "${SAVE_FORMAT:-pt}" \
  --topk "${TOPK:-16}" \
  --seed "${SEED:-2026}" \
  --device "${DEVICE:-cuda}" \
  --use_bf16 "${USE_BF16:-true}" \
  "$@"
