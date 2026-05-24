#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs

# Same defaults as run_round5_expert_input_pos_targeted.sh.
SEQ_LEN="${SEQ_LEN:-128}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-512}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-512}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-513}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-64}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-128}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-2}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-16}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260523}"
RUN_PREFIX="${RUN_PREFIX:-round5-expertpos-targeted}"

MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}"
MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-1}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-128}"
MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-false}"
MOE_ROUTER_TYPE="${MOE_ROUTER_TYPE:-linear}"

short_pos() {
  case "$1" in
    attention_output_residual) echo "resid" ;;
    attention_output) echo "attn" ;;
    layer_input) echo "layerin" ;;
    *) echo "$1" ;;
  esac
}

run_case() {
  local router_pos="$1"
  local router_shape="$2"
  local expert_pos="$3"
  local expert_shape="full"
  local router_short
  local expert_short
  router_short="$(short_pos "${router_pos}")"
  expert_short="$(short_pos "${expert_pos}")"
  local run_name="${RUN_PREFIX}-r${router_shape}-${router_short}-e${expert_short}"
  local log_path="../logs/${run_name}.train.log"

  if [ -f "../checkpoints/${run_name}/${TOTAL_TRAINING_STEPS}.pth" ]; then
    echo "[$(date)] skip ${run_name}: checkpoint exists"
    return
  fi

  echo "[$(date)] start ${run_name}"
  echo "  router=${router_pos}/${router_shape}, expert=${expert_pos}/${expert_shape}"

  env \
    RUN_NAME="${run_name}" \
    DATASET_TYPE="hierarchical_pattern" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE}" \
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
    SEQ_LEN="${SEQ_LEN}" \
    SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}" \
    SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE}" \
    SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
    SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
    SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
    SYNTHETIC_SEED="${SYNTHETIC_SEED}" \
    SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION}" \
    SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA}" \
    SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" \
    DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE}" \
    DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE}" \
    DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE}" \
    DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS}" \
    DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS}" \
    DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS}" \
    DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM}" \
    DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS}" \
    CKPT_DIR="../checkpoints/${run_name}" \
    USE_MOE="true" \
    MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS}" \
    MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK}" \
    MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE}" \
    MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT}" \
    MOE_ROUTER_TYPE="${MOE_ROUTER_TYPE}" \
    MOE_ROUTER_INPUT="hidden" \
    MOE_ROUTER_INPUT_POS="${router_pos}" \
    MOE_ROUTER_INPUT_SHAPE="${router_shape}" \
    MOE_EXPERT_INPUT_POS="${expert_pos}" \
    MOE_EXPERT_INPUT_SHAPE="${expert_shape}" \
    MOE_HEAD_LEVEL="false" \
    LR="${LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

  echo "[$(date)] done ${run_name}, log=${log_path}"
}

echo "Round 5 targeted expert-input-position resume start: $(date)"
echo "steps: ${TOTAL_TRAINING_STEPS}"
echo "remaining cases: 6"

run_case "layer_input" "head" "v"
run_case "layer_input" "head" "hidden"
run_case "q" "head" "attention_output"
run_case "q" "head" "k"
run_case "k" "full" "attention_output"
run_case "layer_input" "full" "attention_output"

echo "Round 5 targeted expert-input-position resume done: $(date)"
