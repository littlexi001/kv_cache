#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Runs 8 ground-truth routing conditions sequentially:
#   distribution: uniform / zipf
#   feature layer: local slot / higher-level unit
#   ground-truth strategy: hash / frequency_balanced
#
# Override shared hyperparameters by exporting them before launching this script.

COMMON_ENV=(
  "LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-16}"
  "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-1000}"
  "SEQ_LEN=${SEQ_LEN:-128}"
  "SYNTHETIC_NUM_SAMPLES=${SYNTHETIC_NUM_SAMPLES:-200000}"
  "SYNTHETIC_BLOCK_SIZE=${SYNTHETIC_BLOCK_SIZE:-4}"
  "SYNTHETIC_NUM_HIERARCHY_LAYERS=${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
  "SYNTHETIC_CONTENT_TOKEN_COUNT=${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
  "SYNTHETIC_NUM_UNITS_PER_LAYER=${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
  "SYNTHETIC_SEED=${SYNTHETIC_SEED:-0}"
  "SYNTHETIC_ZIPF_ALPHA=${SYNTHETIC_ZIPF_ALPHA:-1.0}"
  "SYNTHETIC_ZIPF_SHUFFLE_RANKS=${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"
  "DEBUG_VOCAB_SIZE=${DEBUG_VOCAB_SIZE:-257}"
  "DEBUG_HIDDEN_SIZE=${DEBUG_HIDDEN_SIZE:-128}"
  "DEBUG_INTERMEDIATE_SIZE=${DEBUG_INTERMEDIATE_SIZE:-256}"
  "DEBUG_NUM_HIDDEN_LAYERS=${DEBUG_NUM_HIDDEN_LAYERS:-3}"
  "DEBUG_NUM_ATTENTION_HEADS=${DEBUG_NUM_ATTENTION_HEADS:-4}"
  "DEBUG_NUM_KEY_VALUE_HEADS=${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
  "DEBUG_HEAD_DIM=${DEBUG_HEAD_DIM:-32}"
  "DEBUG_MAX_POSITION_EMBEDDINGS=${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"
  "USE_MOE=true"
  "MOE_NUM_UNIQUE_EXPERTS=${MOE_NUM_UNIQUE_EXPERTS:-4}"
  "MOE_NUM_EXPERTS_PER_TOK=1"
  "MOE_INTERMEDIATE_SIZE=${MOE_INTERMEDIATE_SIZE:-128}"
  "MOE_USE_COMMON_EXPERT=${MOE_USE_COMMON_EXPERT:-false}"
  "MOE_COMMON_INTERMEDIATE_SIZE=${MOE_COMMON_INTERMEDIATE_SIZE:--1}"
  "MOE_ROUTER_INPUT=${MOE_ROUTER_INPUT:-hidden}"
  "MOE_HEAD_LEVEL=false"
  "MOE_LOAD_BALANCE_LOSS_WEIGHT=0.0"
  "GROUND_TRUTH_FREQUENCY_ESTIMATE_SAMPLES=${GROUND_TRUTH_FREQUENCY_ESTIMATE_SAMPLES:-4096}"
  "ATTENTION_STRIDE_PATTERN=${ATTENTION_STRIDE_PATTERN:-1,1,1}"
  "RESIDUAL_SOURCE_PATTERN=${RESIDUAL_SOURCE_PATTERN:--1,-1,-1}"
  "LR=${LR:-1e-3}"
  "WARMUP_STEPS=${WARMUP_STEPS:-100}"
  "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5000}"
  "TRAINING_SEED=${TRAINING_SEED:--1}"
)

for distribution in uniform zipf; do
  for feature in local higher; do
    if [ "$feature" = "local" ]; then
      feature_layer=0
    else
      feature_layer=1
    fi
    for strategy in hash frequency_balanced; do
      run_name="ground-truth-${distribution}-${feature}-${strategy}"
      echo "Launching ${run_name}"
      mkdir -p ../logs
      env "${COMMON_ENV[@]}" \
        "RUN_NAME=${run_name}" \
        "CKPT_DIR=../checkpoints/${run_name}" \
        "SYNTHETIC_SAMPLING_DISTRIBUTION=${distribution}" \
        "GROUND_TRUTH_ROUTING_STRATEGY=${strategy}" \
        "GROUND_TRUTH_ROUTING_FEATURE_LAYER=${feature_layer}" \
        bash single_thread_debug.sh \
        > "../logs/${run_name}.log" 2>&1
    done
  done
done
