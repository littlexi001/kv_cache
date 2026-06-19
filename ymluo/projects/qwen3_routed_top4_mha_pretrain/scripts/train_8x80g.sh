#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-routed_top4_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/routed_top4_qwen3_0p6b_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-${OUTPUT_DIR}/token_cache}"
DATASET_SAMPLE_FILES="${DATASET_SAMPLE_FILES:-1024}"
DATASET_SAMPLE_SEED="${DATASET_SAMPLE_SEED:-1234}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-72000}"
TOKENIZE_MAX_CHARS="${TOKENIZE_MAX_CHARS:-200000000}"
TOKENIZE_MAX_CHARS_PER_FILE="${TOKENIZE_MAX_CHARS_PER_FILE:-250000}"
TOKENIZE_CHUNK_CHARS="${TOKENIZE_CHUNK_CHARS:-2000000}"
CACHE_WAIT_TIMEOUT_SECONDS="${CACHE_WAIT_TIMEOUT_SECONDS:-86400}"
CACHE_POLL_SECONDS="${CACHE_POLL_SECONDS:-5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  "${PROJECT_DIR}/src/train_routed_top4_qwen.py" \
  --model_config_path /mnt/workspace/Qwen3-0.6B/config.json \
  --tokenizer_path /mnt/workspace/Qwen3-0.6B \
  --train_data_root /mnt/workspace/dclm \
  --train_text_glob "*.txt" \
  --dataset_sample_files "${DATASET_SAMPLE_FILES}" \
  --dataset_sample_seed "${DATASET_SAMPLE_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --seq_len 2048 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_steps 1000000 \
  --max_train_seconds "${MAX_TRAIN_SECONDS}" \
  --learning_rate 3e-4 \
  --min_lr_ratio 0.1 \
  --warmup_steps 1000 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --router_top_k 4 \
  --router_aux_loss_coef 0.01 \
  --router_z_loss_coef 0.001 \
  --router_temperature 1.0 \
  --router_noise_std 0.1 \
  --log_steps 10 \
  --save_steps 500 \
  --token_cache_dir "${TOKEN_CACHE_DIR}" \
  --tokenize_max_chars "${TOKENIZE_MAX_CHARS}" \
  --tokenize_max_chars_per_file "${TOKENIZE_MAX_CHARS_PER_FILE}" \
  --tokenize_chunk_chars "${TOKENIZE_CHUNK_CHARS}" \
  --cache_wait_timeout_seconds "${CACHE_WAIT_TIMEOUT_SECONDS}" \
  --cache_poll_seconds "${CACHE_POLL_SECONDS}" \
  "$@"
