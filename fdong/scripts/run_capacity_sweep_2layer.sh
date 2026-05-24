#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs

HIDDEN_SIZES="${HIDDEN_SIZES:-32 48 64 96 128}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}"
RUN_PREFIX="${RUN_PREFIX:-capacity-2layer-${SYNTHETIC_SAMPLING_DISTRIBUTION}}"

echo "Capacity sweep start: $(date)"
echo "hidden sizes: ${HIDDEN_SIZES}"
echo "steps: ${TOTAL_TRAINING_STEPS}"
echo "distribution: ${SYNTHETIC_SAMPLING_DISTRIBUTION}"

for hidden in ${HIDDEN_SIZES}; do
  intermediate=$((hidden * 2))
  head_dim=$((hidden / 4))
  run_name="${RUN_PREFIX}-h${hidden}"
  log_path="../logs/${run_name}.train.log"

  echo "[$(date)] start ${run_name}"

  env \
    RUN_NAME="${run_name}" \
    DATASET_TYPE="hierarchical_pattern" \
    SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}" \
    SEQ_LEN="${SEQ_LEN:-128}" \
    SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}" \
    SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}" \
    SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}" \
    SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}" \
    SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}" \
    SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}" \
    DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-257}" \
    DEBUG_HIDDEN_SIZE="${hidden}" \
    DEBUG_INTERMEDIATE_SIZE="${intermediate}" \
    DEBUG_NUM_HIDDEN_LAYERS="2" \
    DEBUG_NUM_ATTENTION_HEADS="4" \
    DEBUG_NUM_KEY_VALUE_HEADS="2" \
    DEBUG_HEAD_DIM="${head_dim}" \
    DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}" \
    CKPT_DIR="../checkpoints/${run_name}" \
    USE_MOE="${USE_MOE:-false}" \
    MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}" \
    MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-1}" \
    MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-$intermediate}" \
    MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-false}" \
    LR="${LR:-1e-3}" \
    WARMUP_STEPS="${WARMUP_STEPS:-100}" \
    TRAINING_SEED="${TRAINING_SEED:-20260523}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

  echo "[$(date)] done ${run_name}, log=${log_path}"
done

echo "Capacity sweep done: $(date)"
