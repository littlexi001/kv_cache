#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/train_interval_subseq_retrieval.py" \
  --config_dir "${CONFIG_DIR:-/mnt/workspace/Qwen3-0.6B}" \
  --output_dir "${OUT_DIR:-${PROJECT_DIR}/outputs/train}" \
  --run_name "${RUN_NAME:-unet8-small-intervals123-lm}" \
  --total_token "${TOTAL_TOKEN:-10000}" \
  --subseq_len "${SUBSEQ_LEN:-4}" \
  --seq_len "${SEQ_LEN:-1024}" \
  --intervals "${INTERVALS:-1,2,3}" \
  --interval_group_mode "${INTERVAL_GROUP_MODE:-scaled}" \
  --num_hidden_layers "${NUM_HIDDEN_LAYERS:-8}" \
  --attention_stride_pattern "${ATTENTION_STRIDE_PATTERN:-1,1,4,4,4,4,1,1}" \
  --auto_resize_vocab "${AUTO_RESIZE_VOCAB:-true}" \
  --total_steps "${TOTAL_STEPS:-10000}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-1e-4}" \
  --warmup_steps "${WARMUP_STEPS:-200}" \
  --save_interval "${SAVE_INTERVAL:-1000}" \
  --eval_interval "${EVAL_INTERVAL:-100}" \
  --eval_batches "${EVAL_BATCHES:-8}" \
  --train_mode "${TRAIN_MODE:-full_sequence_lm}" \
  --seed "${SEED:-1234}" \
  --device "${DEVICE:-cuda}" \
  --use_bf16 "${USE_BF16:-true}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  "$@"
