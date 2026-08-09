#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load only variables that were not already exported by the caller. This makes
# `GPU_LIST=0,1 bash scripts/run_*.sh` override `.env` as users expect.
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  while IFS='=' read -r key value; do
    key="${key//[[:space:]]/}"
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < "${PROJECT_ROOT}/.env"
fi

: "${MODEL_ROOT:=/mnt/workspace/Qwen3-0.6B}"
: "${DCLM_ROOT:=/mnt/workspace/dclm}"
: "${RUN_ROOT:=/mnt/workspace/pe_pretrain_100b_iclr27}"
: "${GPU_LIST:=0}"
: "${SEQ_LEN:=8192}"
: "${MICRO_BATCH:=1}"
: "${GRAD_ACCUM:=32}"
: "${GLOBAL_BATCH_SIZE:=256}"
: "${TARGET_TOKENS:=100000000000}"
: "${MILESTONE_TOKENS:=100000000,1000000000,10000000000,25000000000,50000000000,75000000000,100000000000}"
: "${TOTAL_STEPS:=47684}"
: "${MILESTONES:=48,477,4769,11921,23842,35763,47684}"
: "${LEARNING_RATE:=1e-4}"
: "${WARMUP_STEPS:=500}"
: "${WEIGHT_DECAY:=0.1}"
: "${ADAM_BETA1:=0.9}"
: "${ADAM_BETA2:=0.95}"
: "${ADAM_EPSILON:=1e-8}"
: "${SEED:=20260808}"
: "${DATA_SEED:=1701}"
: "${NUM_WORKERS:=2}"
: "${TRAIN_FILES:=200000}"
: "${VALIDATION_FILES:=1024}"
: "${EVAL_LENGTHS:=2048,4096,8192}"
: "${RULER_SAMPLES_PER_TASK:=4}"
: "${PPL_BLOCKS:=16}"
: "${RUN_LONGBENCH:=1}"
: "${LONGBENCH_TASKS:=hotpotqa,2wikimqa,multifieldqa_en}"
: "${LONGBENCH_SAMPLES_PER_TASK:=8}"
: "${MAX_NEW_TOKENS:=32}"
: "${ATTN_IMPLEMENTATION:=sdpa}"
: "${DTYPE:=bfloat16}"
: "${PYTHON_BIN:=python}"
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${TOKENIZERS_PARALLELISM:=false}"
: "${INITIALIZATION:=from_scratch}"
: "${LOGGING_STEPS:=5}"
: "${TENSORBOARD_ENABLED:=1}"
: "${TENSORBOARD_HOST:=0.0.0.0}"
: "${TENSORBOARD_PORT:=6006}"

export MODEL_ROOT DCLM_ROOT RUN_ROOT GPU_LIST
export SEQ_LEN MICRO_BATCH GRAD_ACCUM GLOBAL_BATCH_SIZE TARGET_TOKENS MILESTONE_TOKENS
export TOTAL_STEPS MILESTONES LEARNING_RATE WARMUP_STEPS WEIGHT_DECAY
export ADAM_BETA1 ADAM_BETA2 ADAM_EPSILON SEED DATA_SEED NUM_WORKERS
export TRAIN_FILES VALIDATION_FILES EVAL_LENGTHS RULER_SAMPLES_PER_TASK PPL_BLOCKS
export RUN_LONGBENCH LONGBENCH_TASKS LONGBENCH_SAMPLES_PER_TASK MAX_NEW_TOKENS
export ATTN_IMPLEMENTATION DTYPE PYTHON_BIN HF_ENDPOINT TOKENIZERS_PARALLELISM
export INITIALIZATION LOGGING_STEPS TENSORBOARD_ENABLED TENSORBOARD_HOST TENSORBOARD_PORT

strategy_path() {
  printf '%s/configs/strategies/%s.json' "${PROJECT_ROOT}" "$1"
}

run_dir_for() {
  if [[ "$1" == "base_eval" ]]; then
    printf '%s/base_eval' "${RUN_ROOT}"
  else
    printf '%s/%s' "${RUN_ROOT}" "$1"
  fi
}
