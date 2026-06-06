#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints ../experiments

HIDDEN_SIZES="${HIDDEN_SIZES:-64 96}"
SOURCE_RUN_PREFIX="${SOURCE_RUN_PREFIX:-frequency-width-dense}"
RUN_PREFIX="${RUN_PREFIX:-frequency-width-zipf-to-uniform}"
SOURCE_STEP="${SOURCE_STEP:-1000}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
SEQ_LEN="${SEQ_LEN:-128}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
TRAINING_SEED="${TRAINING_SEED:-20260606}"

SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.3}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

train_one() {
  local hidden="$1"
  local intermediate=$((hidden * 2))
  local head_dim=$((hidden / 4))
  local source_run="${SOURCE_RUN_PREFIX}-zipf-h${hidden}"
  local init_checkpoint="../checkpoints/${source_run}/${SOURCE_STEP}.pth"
  local run_name="${RUN_PREFIX}-h${hidden}"
  local log_path="../logs/${run_name}.train.log"

  if [ ! -f "${init_checkpoint}" ]; then
    echo "missing init checkpoint: ${init_checkpoint}" >&2
    exit 1
  fi

  echo "[$(date)] fine-tune ${run_name} from ${init_checkpoint}"
  env \
    INIT_CHECKPOINT="${init_checkpoint}" \
    DATASET_TYPE="hierarchical_pattern" \
    SYNTHETIC_SAMPLING_DISTRIBUTION="uniform" \
    SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA}" \
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

echo "Frequency-width zipf-to-uniform fine-tune start: $(date)"
for hidden in ${HIDDEN_SIZES}; do
  train_one "${hidden}"
done
echo "Frequency-width zipf-to-uniform fine-tune done: $(date)"
