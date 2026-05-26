#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints

# Round6 controlled reused-token dataset.
SEQ_LEN="${SEQ_LEN:-128}"
SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-512}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-512}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"

CONTROLLED_SAME_INPUT_DIFF_OUTPUT_RATE="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_RATE:-0.1}"
CONTROLLED_SAME_INPUT_DIFF_OUTPUT_SIZE="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_SIZE:-4}"
CONTROLLED_SAME_INPUT_DIFF_OUTPUT_DISTRIBUTION="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_DISTRIBUTION:-zipf}" # uniform/zipf
CONTROLLED_SAME_INPUT_DIFF_OUTPUT_ZIPF_ALPHA="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_ZIPF_ALPHA:-1.0}"
CONTROLLED_DIFF_INPUT_SAME_OUTPUT_RATE="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_RATE:-0.1}"
CONTROLLED_DIFF_INPUT_SAME_OUTPUT_SIZE="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_SIZE:-4}"
CONTROLLED_DIFF_INPUT_SAME_OUTPUT_DISTRIBUTION="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_DISTRIBUTION:-zipf}" # uniform/zipf
CONTROLLED_DIFF_INPUT_SAME_OUTPUT_ZIPF_ALPHA="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_ZIPF_ALPHA:-1.0}"
CONTROLLED_TOP_SAMPLING_DISTRIBUTION="${CONTROLLED_TOP_SAMPLING_DISTRIBUTION:-uniform}" # uniform/zipf
CONTROLLED_TOP_SAMPLING_ZIPF_ALPHA="${CONTROLLED_TOP_SAMPLING_ZIPF_ALPHA:-1.0}"

# Default model / training setting.
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-2}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"
DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-513}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260525}"
RUN_PREFIX="${RUN_PREFIX:-round6-controlled}"

short_pos() {
  case "$1" in
    attention_output_residual) echo "resid" ;;
    attention_output) echo "attn" ;;
    layer_input) echo "layerin" ;;
    *) echo "$1" ;;
  esac
}

run_case() {
  local suffix="$1"
  local use_moe="$2"
  local hidden_size="$3"
  local intermediate_size="$4"
  local head_dim="$5"
  local router_pos="${6:-hidden}"
  local router_shape="${7:-full}"
  local expert_pos="${8:-attention_output_residual}"
  local expert_shape="${9:-full}"
  local num_experts="${10:-4}"
  local topk="${11:-1}"
  local moe_intermediate="${12:-128}"
  local common="${13:-false}"
  local load_balance="${14:-0.0}"

  local run_name="${RUN_PREFIX}-${suffix}"
  local ckpt_path="../checkpoints/${run_name}/${TOTAL_TRAINING_STEPS}.pth"
  local log_path="../logs/${run_name}.train.log"

  if [ -f "${ckpt_path}" ]; then
    echo "[$(date)] skip ${run_name}: found ${ckpt_path}"
    return 0
  fi

  echo "[$(date)] start ${run_name}"
  echo "  use_moe=${use_moe}, h=${hidden_size}, inter=${intermediate_size}, router=${router_pos}/${router_shape}, expert=${expert_pos}/${expert_shape}, experts=${num_experts}, topk=${topk}, common=${common}, lb=${load_balance}"

  env \
    RUN_NAME="${run_name}" \
    DATASET_TYPE="controlled_reused_token" \
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
    CONTROLLED_SAME_INPUT_DIFF_OUTPUT_RATE="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_RATE}" \
    CONTROLLED_SAME_INPUT_DIFF_OUTPUT_SIZE="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_SIZE}" \
    CONTROLLED_SAME_INPUT_DIFF_OUTPUT_DISTRIBUTION="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_DISTRIBUTION}" \
    CONTROLLED_SAME_INPUT_DIFF_OUTPUT_ZIPF_ALPHA="${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_ZIPF_ALPHA}" \
    CONTROLLED_DIFF_INPUT_SAME_OUTPUT_RATE="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_RATE}" \
    CONTROLLED_DIFF_INPUT_SAME_OUTPUT_SIZE="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_SIZE}" \
    CONTROLLED_DIFF_INPUT_SAME_OUTPUT_DISTRIBUTION="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_DISTRIBUTION}" \
    CONTROLLED_DIFF_INPUT_SAME_OUTPUT_ZIPF_ALPHA="${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_ZIPF_ALPHA}" \
    CONTROLLED_TOP_SAMPLING_DISTRIBUTION="${CONTROLLED_TOP_SAMPLING_DISTRIBUTION}" \
    CONTROLLED_TOP_SAMPLING_ZIPF_ALPHA="${CONTROLLED_TOP_SAMPLING_ZIPF_ALPHA}" \
    DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE}" \
    DEBUG_HIDDEN_SIZE="${hidden_size}" \
    DEBUG_INTERMEDIATE_SIZE="${intermediate_size}" \
    DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS}" \
    DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS}" \
    DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS}" \
    DEBUG_HEAD_DIM="${head_dim}" \
    DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS}" \
    CKPT_DIR="../checkpoints/${run_name}" \
    USE_MOE="${use_moe}" \
    MOE_NUM_UNIQUE_EXPERTS="${num_experts}" \
    MOE_NUM_EXPERTS_PER_TOK="${topk}" \
    MOE_INTERMEDIATE_SIZE="${moe_intermediate}" \
    MOE_USE_COMMON_EXPERT="${common}" \
    MOE_ROUTER_TYPE="linear" \
    MOE_ROUTER_INPUT="hidden" \
    MOE_ROUTER_INPUT_POS="${router_pos}" \
    MOE_ROUTER_INPUT_SHAPE="${router_shape}" \
    MOE_EXPERT_INPUT_POS="${expert_pos}" \
    MOE_EXPERT_INPUT_SHAPE="${expert_shape}" \
    MOE_LOAD_BALANCE_LOSS_WEIGHT="${load_balance}" \
    LR="${LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

  echo "[$(date)] done ${run_name}, log=${log_path}"
}

echo "Round6 controlled reused-token broad sweep start: $(date)"
echo "data: slot=${SYNTHETIC_BLOCK_SIZE}, layers=${SYNTHETIC_NUM_HIERARCHY_LAYERS}, units=${SYNTHETIC_NUM_UNITS_PER_LAYER}, content=${SYNTHETIC_CONTENT_TOKEN_COUNT}"
echo "A same-input-diff-output: rate=${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_RATE}, size=${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_SIZE}, dist=${CONTROLLED_SAME_INPUT_DIFF_OUTPUT_DISTRIBUTION}"
echo "B diff-input-same-output: rate=${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_RATE}, size=${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_SIZE}, dist=${CONTROLLED_DIFF_INPUT_SAME_OUTPUT_DISTRIBUTION}"
echo "steps=${TOTAL_TRAINING_STEPS}"

echo
echo "== 1. Dense capacity baselines =="
run_case "dense-h32" false 32 64 8
run_case "dense-h64" false 64 128 16
run_case "dense-h128" false 128 256 32

echo
echo "== 2. Ordinary token-level MoE baselines =="
run_case "moe-hidden-full-top1" true 64 128 16 hidden full attention_output_residual full 4 1 128 false 0.0
run_case "moe-hidden-full-top2" true 64 128 16 hidden full attention_output_residual full 4 2 128 false 0.0
run_case "moe-hidden-full-common" true 64 128 16 hidden full attention_output_residual full 4 1 128 true 0.0
run_case "moe-hidden-full-lb001" true 64 128 16 hidden full attention_output_residual full 4 1 128 false 0.01

echo
echo "== 3. Router input / router shape sweep with full residual expert input =="
for router_shape in full head; do
  for router_pos in attention_output q k v layer_input hidden; do
    router_short="$(short_pos "${router_pos}")"
    run_case "moe-r${router_shape}-${router_short}-eresid" true 64 128 16 "${router_pos}" "${router_shape}" attention_output_residual full 4 1 128 false 0.0
  done
done

echo
echo "== 4. k/head capacity and regularization =="
for num_experts in 4 8; do
  for topk in 1 2; do
    run_case "moe-k-head-ne${num_experts}-topk${topk}" true 64 128 16 k head attention_output_residual full "${num_experts}" "${topk}" 128 false 0.0
  done
done
run_case "moe-k-head-common" true 64 128 16 k head attention_output_residual full 4 1 128 true 0.0
run_case "moe-k-head-lb001" true 64 128 16 k head attention_output_residual full 4 1 128 false 0.01
run_case "moe-k-head-lb01" true 64 128 16 k head attention_output_residual full 4 1 128 false 0.1

echo
echo "== 5. Expert input ablation for k/head router =="
for expert_pos in attention_output hidden layer_input q k v; do
  expert_short="$(short_pos "${expert_pos}")"
  run_case "moe-k-head-e${expert_short}" true 64 128 16 k head "${expert_pos}" full 4 1 128 false 0.0
done

echo
echo "== 6. True head/head ablation =="
for expert_pos in attention_output_residual attention_output q k v layer_input hidden; do
  expert_short="$(short_pos "${expert_pos}")"
  run_case "moe-headhead-k-e${expert_short}" true 64 128 16 k head "${expert_pos}" head 4 1 32 false 0.0
done

echo "Round6 controlled reused-token broad sweep done: $(date)"
