#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen1.5-MoE-A2.7B}"
RUN_NAME="${RUN_NAME:-qwen15-moe-svd-projected-routing}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/train/${RUN_NAME}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
  "${PROJECT_DIR}/src/train_svd_projected_routing.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --data_mode "${DATA_MODE:-synthetic}" \
  --dataset_path "${DATA_PATH:-/mnt/workspace/dclm}" \
  --model_size_preset "${MODEL_SIZE_PRESET:-moe_0_6b}" \
  --model_config_overrides "${MODEL_CONFIG_OVERRIDES:-}" \
  --projection_source "${PROJECTION_SOURCE:-q}" \
  --svd_refresh_interval "${SVD_REFRESH_INTERVAL:-100}" \
  --group1_experts "${GROUP1_EXPERTS:-16}" \
  --group2_experts "${GROUP2_EXPERTS:-24}" \
  --group3_experts "${GROUP3_EXPERTS:-8}" \
  --group1_topk "${GROUP1_TOPK:-2}" \
  --group2_topk "${GROUP2_TOPK:-3}" \
  --group3_topk "${GROUP3_TOPK:-1}" \
  --seq_length "${SEQ_LENGTH:-256}" \
  --synthetic_vocab_size "${SYNTHETIC_VOCAB_SIZE:-4096}" \
  --synthetic_topic_count "${SYNTHETIC_TOPIC_COUNT:-16}" \
  --synthetic_entities_per_topic "${SYNTHETIC_ENTITIES_PER_TOPIC:-32}" \
  --synthetic_noise_rate "${SYNTHETIC_NOISE_RATE:-0.15}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --max_steps "${MAX_STEPS:-10000}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 "${BF16:-true}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-false}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-true}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --load_balance_loss_weight "${LOAD_BALANCE_LOSS_WEIGHT:-0.01}" \
  --deepspeed_config "${DEEPSPEED_CONFIG:-}" \
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-}" \
  --report_to "${REPORT_TO:-tensorboard}"
