#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs

# Final Round 5 data setting.
SEQ_LEN="${SEQ_LEN:-128}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-512}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-512}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

# Fixed small model setting.
DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-513}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-64}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-128}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-2}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-16}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

# Training setting.
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260523}"
RUN_PREFIX="${RUN_PREFIX:-round5}"

# Baseline MoE capacity. Spectral cases override the expert layout but keep
# the same active intermediate budget: 32 + 48 + 48 = 128.
MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}"
MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-1}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-128}"
MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-false}"
MOE_ROUTER_TYPE="${MOE_ROUTER_TYPE:-linear}"

run_case() {
  local suffix="$1"
  local router_pos="$2"
  local router_shape="$3"
  local expert_pos="$4"
  local expert_shape="$5"
  local is_head="${6:-false}"
  local is_spectral="${7:-false}"

  local run_name="${RUN_PREFIX}-${suffix}"
  local log_path="../logs/${run_name}.train.log"

  echo "[$(date)] start ${run_name}"
  echo "  router=${router_pos}/${router_shape}, expert=${expert_pos}/${expert_shape}, head=${is_head}, spectral=${is_spectral}"

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
    MOE_HEAD_LEVEL="${is_head}" \
    MOE_SPECTRAL_BAND_DIMS="$([ "${is_spectral}" = "true" ] && echo "8,32,64" || echo "")" \
    MOE_SPECTRAL_NUM_EXPERTS_PER_BAND="$([ "${is_spectral}" = "true" ] && echo "0,4,4" || echo "")" \
    MOE_SPECTRAL_TOPK_PER_BAND="$([ "${is_spectral}" = "true" ] && echo "1,1,1" || echo "")" \
    MOE_SPECTRAL_INTERMEDIATE_SIZES="$([ "${is_spectral}" = "true" ] && echo "32,48,48" || echo "")" \
    MOE_SPECTRAL_WARMUP_STEPS="${MOE_SPECTRAL_WARMUP_STEPS:-100}" \
    MOE_SPECTRAL_UPDATE_INTERVAL="${MOE_SPECTRAL_UPDATE_INTERVAL:-100}" \
    MOE_SPECTRAL_SAMPLE_SIZE="${MOE_SPECTRAL_SAMPLE_SIZE:-4096}" \
    MOE_SPECTRAL_BASIS_MOMENTUM="${MOE_SPECTRAL_BASIS_MOMENTUM:-0.0}" \
    LR="${LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

  echo "[$(date)] done ${run_name}, log=${log_path}"
}

echo "Round 5 MoE structure sweep start: $(date)"
echo "data: b${SYNTHETIC_BLOCK_SIZE}-u${SYNTHETIC_NUM_UNITS_PER_LAYER}-vocab${SYNTHETIC_CONTENT_TOKEN_COUNT}-${SYNTHETIC_SAMPLING_DISTRIBUTION}${SYNTHETIC_ZIPF_ALPHA}"
echo "model: layers=${DEBUG_NUM_HIDDEN_LAYERS}, hidden=${DEBUG_HIDDEN_SIZE}, intermediate=${DEBUG_INTERMEDIATE_SIZE}"
echo "steps: ${TOTAL_TRAINING_STEPS}"

run_case "full-attn-output-exp-resid" "attention_output" "full" "attention_output_residual" "full"
run_case "full-q-exp-resid" "q" "full" "attention_output_residual" "full"
run_case "full-k-exp-resid" "k" "full" "attention_output_residual" "full"
run_case "full-v-exp-resid" "v" "full" "attention_output_residual" "full"
run_case "full-layer-input-exp-resid" "layer_input" "full" "attention_output_residual" "full"
run_case "full-hidden-exp-resid" "hidden" "full" "attention_output_residual" "full"
run_case "head-attn-output-exp-resid" "attention_output" "head" "attention_output_residual" "full"
run_case "head-q-exp-resid" "q" "head" "attention_output_residual" "full"
run_case "head-k-exp-resid" "k" "head" "attention_output_residual" "full"
run_case "head-v-exp-resid" "v" "head" "attention_output_residual" "full"
run_case "head-layer-input-exp-resid" "layer_input" "head" "attention_output_residual" "full"
run_case "head-hidden-exp-resid" "hidden" "head" "attention_output_residual" "full"
run_case "spectral-attn-output-exp-resid" "attention_output" "spectral" "attention_output_residual" "full" "false" "true"
run_case "spectral-q-exp-resid" "q" "spectral" "attention_output_residual" "full" "false" "true"
run_case "spectral-k-exp-resid" "k" "spectral" "attention_output_residual" "full" "false" "true"
run_case "spectral-v-exp-resid" "v" "spectral" "attention_output_residual" "full" "false" "true"
run_case "spectral-layer-input-exp-resid" "layer_input" "spectral" "attention_output_residual" "full" "false" "true"
run_case "spectral-hidden-exp-resid" "hidden" "spectral" "attention_output_residual" "full" "false" "true"

echo "Round 5 MoE structure sweep done: $(date)"
