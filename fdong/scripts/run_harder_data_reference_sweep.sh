#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs

# Fixed model: dense 2-layer h64. This sweep changes only synthetic data difficulty.
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-128}"
HEAD_DIM="${HEAD_DIM:-16}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SEQ_LEN="${SEQ_LEN:-128}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260523}"

# User-chosen fixed data settings.
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-512}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

# Difficulty sweep: larger slot length and more candidate patterns per layer.
BLOCK_SIZES="${BLOCK_SIZES:-6 8}"
UNITS_PER_LAYER_VALUES="${UNITS_PER_LAYER_VALUES:-128 256}"

RUN_PREFIX="${RUN_PREFIX:-harder-data-ref}"

echo "Harder-data reference sweep start: $(date)"
echo "model: dense 2-layer h${HIDDEN_SIZE}"
echo "steps: ${TOTAL_TRAINING_STEPS}"
echo "seq_len: ${SEQ_LEN}"
echo "hierarchy_layers: ${SYNTHETIC_NUM_HIERARCHY_LAYERS}"
echo "content_token_count: ${SYNTHETIC_CONTENT_TOKEN_COUNT}"
echo "sampling: ${SYNTHETIC_SAMPLING_DISTRIBUTION}, zipf_alpha=${SYNTHETIC_ZIPF_ALPHA}"
echo "block sizes: ${BLOCK_SIZES}"
echo "units/layer values: ${UNITS_PER_LAYER_VALUES}"

for block_size in ${BLOCK_SIZES}; do
  for units_per_layer in ${UNITS_PER_LAYER_VALUES}; do
    run_name="${RUN_PREFIX}-b${block_size}-u${units_per_layer}"
    log_path="../logs/${run_name}.train.log"

    echo "[$(date)] start ${run_name}"

    env \
      RUN_NAME="${run_name}" \
      DATASET_TYPE="hierarchical_pattern" \
      TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}" \
      GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
      SEQ_LEN="${SEQ_LEN}" \
      SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}" \
      SYNTHETIC_BLOCK_SIZE="${block_size}" \
      SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
      SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
      SYNTHETIC_NUM_UNITS_PER_LAYER="${units_per_layer}" \
      SYNTHETIC_SEED="${SYNTHETIC_SEED}" \
      SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION}" \
      SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA}" \
      SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" \
      DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-513}" \
      DEBUG_HIDDEN_SIZE="${HIDDEN_SIZE}" \
      DEBUG_INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE}" \
      DEBUG_NUM_HIDDEN_LAYERS="2" \
      DEBUG_NUM_ATTENTION_HEADS="4" \
      DEBUG_NUM_KEY_VALUE_HEADS="2" \
      DEBUG_HEAD_DIM="${HEAD_DIM}" \
      DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}" \
      CKPT_DIR="../checkpoints/${run_name}" \
      USE_MOE="false" \
      LR="${LR}" \
      WARMUP_STEPS="${WARMUP_STEPS}" \
      TRAINING_SEED="${TRAINING_SEED}" \
      bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

    echo "[$(date)] done ${run_name}, log=${log_path}"
  done
done

echo "Harder-data reference sweep done: $(date)"
