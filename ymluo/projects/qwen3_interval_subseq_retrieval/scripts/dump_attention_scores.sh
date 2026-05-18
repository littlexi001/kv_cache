#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

ARGS=(
  --run_dir "${RUN_DIR:-${PROJECT_DIR}/outputs/train/${RUN_NAME:-unet8-small-intervals123-lm-ddp}}"
  --ckpt_step "${CKPT_STEP:-2000}"
  --config_dir "${CONFIG_DIR:-/mnt/workspace/Qwen3-0.6B}"
  --batch_size "${BATCH_SIZE:-1}"
  --query_positions "${QUERY_POSITIONS:-all}"
  --save_raw_scores "${SAVE_RAW_SCORES:-true}"
  --save_probabilities "${SAVE_PROBABILITIES:-true}"
  --save_format "${SAVE_FORMAT:-pt}"
  --topk "${TOPK:-16}"
  --seed "${SEED:-2026}"
  --device "${DEVICE:-cuda}"
  --use_bf16 "${USE_BF16:-true}"
  --disable_sliding_window "${DISABLE_SLIDING_WINDOW:-true}"
)

[[ -n "${TOTAL_TOKEN:-}" ]] && ARGS+=(--total_token "${TOTAL_TOKEN}")
[[ -n "${SUBSEQ_LEN:-}" ]] && ARGS+=(--subseq_len "${SUBSEQ_LEN}")
[[ -n "${SEQ_LEN:-}" ]] && ARGS+=(--seq_len "${SEQ_LEN}")
[[ -n "${INTERVALS:-}" ]] && ARGS+=(--intervals "${INTERVALS}")
[[ -n "${INTERVAL_GROUP_MODE:-}" ]] && ARGS+=(--interval_group_mode "${INTERVAL_GROUP_MODE}")
[[ -n "${NUM_HIDDEN_LAYERS:-}" ]] && ARGS+=(--num_hidden_layers "${NUM_HIDDEN_LAYERS}")
[[ -n "${ATTENTION_STRIDE_PATTERN:-}" ]] && ARGS+=(--attention_stride_pattern "${ATTENTION_STRIDE_PATTERN}")

python "${PROJECT_DIR}/src/dump_attention_scores.py" "${ARGS[@]}" "$@"
