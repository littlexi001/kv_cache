#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-kv_head_gate_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_kv_head_gate_training/output/kv_head_gate_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
RESUME_FROM="${RESUME_FROM:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume_from "${RESUME_FROM}")
fi

STREAM_SHUFFLE_ARGS=(--stream_shuffle_files)
if [[ "${STREAM_SHUFFLE_FILES:-true}" == "0" || "${STREAM_SHUFFLE_FILES:-true}" == "false" || "${STREAM_SHUFFLE_FILES:-true}" == "False" ]]; then
  STREAM_SHUFFLE_ARGS=(--no-stream_shuffle_files)
fi

TRAIN_BASE_ARGS=(--train_base_model)
if [[ "${TRAIN_BASE_MODEL:-true}" == "0" || "${TRAIN_BASE_MODEL:-true}" == "false" || "${TRAIN_BASE_MODEL:-true}" == "False" ]]; then
  TRAIN_BASE_ARGS=(--no-train_base_model)
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "${PROJECT_DIR}/src/train_kv_head_gate_qwen3.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH:-/mnt/workspace/Qwen3-0.6B}" \
  --train_data_root "${TRAIN_DATA_ROOT:-/mnt/workspace/dclm}" \
  --train_text_glob "${TRAIN_TEXT_GLOB:-*.txt}" \
  --output_dir "${OUTPUT_DIR}" \
  --seq_len "${SEQ_LEN:-2048}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --max_steps "${MAX_STEPS:-1000000}" \
  --max_train_seconds "${MAX_TRAIN_SECONDS:-72000}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --gate_learning_rate "${GATE_LEARNING_RATE:-1e-4}" \
  --min_lr_ratio "${MIN_LR_RATIO:-0.1}" \
  --warmup_steps "${WARMUP_STEPS:-500}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --target_keep_ratio "${TARGET_KEEP_RATIO:-0.20}" \
  --gate_hard_mode "${GATE_HARD_MODE:-global_budget}" \
  --gate_threshold "${GATE_THRESHOLD:-0.5}" \
  --gate_temperature "${GATE_TEMPERATURE:-1.0}" \
  --gate_sink_tokens_all_heads "${GATE_SINK_TOKENS_ALL_HEADS:-64}" \
  --budget_loss_coef "${BUDGET_LOSS_COEF:-0.05}" \
  --load_loss_coef "${LOAD_LOSS_COEF:-0.01}" \
  --z_loss_coef "${Z_LOSS_COEF:-0.001}" \
  "${TRAIN_BASE_ARGS[@]}" \
  "${STREAM_SHUFFLE_ARGS[@]}" \
  --stream_max_files_per_rank_epoch "${STREAM_MAX_FILES_PER_RANK_EPOCH:-0}" \
  --stream_chunk_chars "${STREAM_CHUNK_CHARS:-2000000}" \
  --stream_max_chars_per_file "${STREAM_MAX_CHARS_PER_FILE:-0}" \
  --log_steps "${LOG_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --seed "${SEED:-1234}" \
  "${RESUME_ARGS[@]}" \
  "$@"
