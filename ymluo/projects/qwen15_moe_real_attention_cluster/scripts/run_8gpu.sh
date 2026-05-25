#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen1.5-MoE-A2.7B}"
DATA_PATH="${DATA_PATH:-/mnt/workspace/dclm}"
MODEL_SIZE_PRESET="${MODEL_SIZE_PRESET:-moe_0_6b}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-attention_cluster}"
if [[ -z "${OUT_DIR+x}" ]]; then
  if [[ "${EXPERIMENT_MODE}" == "baseline" ]]; then
    OUT_DIR="${PROJECT_DIR}/outputs/qwen15-moe-0p6b-baseline"
  else
    OUT_DIR="${PROJECT_DIR}/outputs/qwen15-moe-0p6b-real-attn-cluster"
  fi
fi
if [[ -z "${RUN_NAME+x}" ]]; then
  if [[ "${EXPERIMENT_MODE}" == "baseline" ]]; then
    RUN_NAME="qwen15-moe-0p6b-baseline"
  else
    RUN_NAME="qwen15-moe-0p6b-real-attn-cluster"
  fi
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 40000))}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
if [[ -z "${DEEPSPEED_CONFIG+x}" ]]; then
  if [[ "${MODEL_SIZE_PRESET}" == "none" ]]; then
    DEEPSPEED_CONFIG="${PROJECT_DIR}/configs/deepspeed_zero3.json"
  else
    DEEPSPEED_CONFIG=""
  fi
fi

EXTRA_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  EXTRA_ARGS+=(--deepspeed_config "${DEEPSPEED_CONFIG}")
fi
if [[ -n "${MODEL_CONFIG_OVERRIDES:-}" ]]; then
  EXTRA_ARGS+=(--model_config_overrides "${MODEL_CONFIG_OVERRIDES}")
fi

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/src/train_real_attention_cluster.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATA_PATH}" \
  --data_files_glob "${DATA_FILES_GLOB:-**/*.txt}" \
  --output_dir "${OUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --experiment_mode "${EXPERIMENT_MODE}" \
  --init_from_scratch "${INIT_FROM_SCRATCH:-true}" \
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-}" \
  --model_size_preset "${MODEL_SIZE_PRESET}" \
  --seed "${SEED:-1234}" \
  --seq_length "${SEQ_LENGTH:-1024}" \
  --min_text_chars "${MIN_TEXT_CHARS:-20}" \
  --attention_top_ratio "${ATTENTION_TOP_RATIO:-0.10}" \
  --expert_input_top_ratio "${EXPERT_INPUT_TOP_RATIO:-0.10}" \
  --include_self "${INCLUDE_SELF:-false}" \
  --attention_cluster_weight "${ATTENTION_CLUSTER_WEIGHT:-0.01}" \
  --attention_cluster_temperature "${ATTENTION_CLUSTER_TEMPERATURE:-1.0}" \
  --attention_cluster_detach_attention "${ATTENTION_CLUSTER_DETACH_ATTENTION:-true}" \
  --attention_cluster_detach_key_router "${ATTENTION_CLUSTER_DETACH_KEY_ROUTER:-false}" \
  --load_balance_loss_weight "${LOAD_BALANCE_LOSS_WEIGHT:-0.01}" \
  --load_balance_temperature "${LOAD_BALANCE_TEMPERATURE:-1.0}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --max_steps "${MAX_STEPS:-10000}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 "${BF16:-true}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-true}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-false}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --report_to "${REPORT_TO:-tensorboard}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
