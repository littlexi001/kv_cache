#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints ../experiments

HIDDEN_SIZES="${HIDDEN_SIZES:-64 96}"
ZIPF_ALPHAS="${ZIPF_ALPHAS:-0.7 1.0 1.3 1.6}"
RUN_PREFIX="${RUN_PREFIX:-frequency-width-skew}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SEQ_LEN="${SEQ_LEN:-128}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260605}"

SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

train_one() {
  local alpha="$1"
  local hidden="$2"
  local alpha_tag
  alpha_tag="$(printf "%s" "${alpha}" | tr '.' 'p')"
  local intermediate=$((hidden * 2))
  local head_dim=$((hidden / 4))
  local run_name="${RUN_PREFIX}-zipf${alpha_tag}-h${hidden}"
  local log_path="../logs/${run_name}.train.log"

  echo "[$(date)] train ${run_name}"
  env \
    DATASET_TYPE="hierarchical_pattern" \
    SYNTHETIC_SAMPLING_DISTRIBUTION="zipf" \
    SYNTHETIC_ZIPF_ALPHA="${alpha}" \
    SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    SEQ_LEN="${SEQ_LEN}" \
    SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES}" \
    SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE}" \
    SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
    SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
    SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
    SYNTHETIC_SEED="${SYNTHETIC_SEED}" \
    DEBUG_VOCAB_SIZE="$((SYNTHETIC_CONTENT_TOKEN_COUNT + 1))" \
    DEBUG_HIDDEN_SIZE="${hidden}" \
    DEBUG_INTERMEDIATE_SIZE="${intermediate}" \
    DEBUG_NUM_HIDDEN_LAYERS="2" \
    DEBUG_NUM_ATTENTION_HEADS="4" \
    DEBUG_NUM_KEY_VALUE_HEADS="2" \
    DEBUG_HEAD_DIM="${head_dim}" \
    DEBUG_MAX_POSITION_EMBEDDINGS="256" \
    USE_MOE="false" \
    MOE_NUM_UNIQUE_EXPERTS="4" \
    MOE_NUM_EXPERTS_PER_TOK="1" \
    MOE_INTERMEDIATE_SIZE="${intermediate}" \
    MOE_USE_COMMON_EXPERT="false" \
    LR="${LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    CKPT_DIR="../checkpoints/${run_name}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1
  echo "[$(date)] done ${run_name}, log=${log_path}"
}

echo "Frequency-width skew sweep start: $(date)"
for alpha in ${ZIPF_ALPHAS}; do
  for hidden in ${HIDDEN_SIZES}; do
    train_one "${alpha}" "${hidden}"
  done
done
echo "Frequency-width skew sweep done: $(date)"
