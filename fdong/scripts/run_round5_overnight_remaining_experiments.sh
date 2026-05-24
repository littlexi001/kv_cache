#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints

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
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TRAINING_SEED="${TRAINING_SEED:-20260523}"

ROUTER_POSITIONS=(attention_output q k v layer_input hidden)
EXPERT_POSITIONS=(attention_output_residual attention_output layer_input q k v hidden)

short_pos() {
  case "$1" in
    attention_output_residual) echo "resid" ;;
    attention_output) echo "attn" ;;
    layer_input) echo "layerin" ;;
    *) echo "$1" ;;
  esac
}

run_case() {
  local run_name="$1"
  local router_pos="$2"
  local router_shape="$3"
  local expert_pos="$4"
  local expert_shape="$5"
  local moe_intermediate_size="$6"
  local num_experts="$7"
  local topk="$8"
  local use_common="$9"
  local spectral_bands="${10:-}"
  local spectral_experts="${11:-}"
  local spectral_topk="${12:-}"
  local spectral_intermediates="${13:-}"
  local spectral_sample_size="${14:-4096}"
  local spectral_warmup_steps="${15:-100}"
  local spectral_basis_momentum="${16:-0.0}"

  local ckpt_path="../checkpoints/${run_name}/${TOTAL_TRAINING_STEPS}.pth"
  local log_path="../logs/${run_name}.train.log"
  if [ -f "${ckpt_path}" ]; then
    echo "[$(date)] skip ${run_name}: found ${ckpt_path}"
    return 0
  fi

  echo "[$(date)] start ${run_name}"
  echo "  router=${router_pos}/${router_shape}, expert=${expert_pos}/${expert_shape}, experts=${num_experts}, topk=${topk}, common=${use_common}, moe_intermediate=${moe_intermediate_size}"

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
    MOE_NUM_UNIQUE_EXPERTS="${num_experts}" \
    MOE_NUM_EXPERTS_PER_TOK="${topk}" \
    MOE_INTERMEDIATE_SIZE="${moe_intermediate_size}" \
    MOE_USE_COMMON_EXPERT="${use_common}" \
    MOE_ROUTER_TYPE="linear" \
    MOE_ROUTER_INPUT="hidden" \
    MOE_ROUTER_INPUT_POS="${router_pos}" \
    MOE_ROUTER_INPUT_SHAPE="${router_shape}" \
    MOE_EXPERT_INPUT_POS="${expert_pos}" \
    MOE_EXPERT_INPUT_SHAPE="${expert_shape}" \
    MOE_HEAD_LEVEL="false" \
    MOE_SPECTRAL_BAND_DIMS="${spectral_bands}" \
    MOE_SPECTRAL_NUM_EXPERTS_PER_BAND="${spectral_experts}" \
    MOE_SPECTRAL_TOPK_PER_BAND="${spectral_topk}" \
    MOE_SPECTRAL_INTERMEDIATE_SIZES="${spectral_intermediates}" \
    MOE_SPECTRAL_WARMUP_STEPS="${spectral_warmup_steps}" \
    MOE_SPECTRAL_UPDATE_INTERVAL="${MOE_SPECTRAL_UPDATE_INTERVAL:-100}" \
    MOE_SPECTRAL_SAMPLE_SIZE="${spectral_sample_size}" \
    MOE_SPECTRAL_BASIS_MOMENTUM="${spectral_basis_momentum}" \
    LR="${LR}" \
    WARMUP_STEPS="${WARMUP_STEPS}" \
    TRAINING_SEED="${TRAINING_SEED}" \
    bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1

  echo "[$(date)] done ${run_name}, log=${log_path}"
}

echo "Round 5 overnight experiments start: $(date)"
echo "data: b${SYNTHETIC_BLOCK_SIZE}-u${SYNTHETIC_NUM_UNITS_PER_LAYER}-vocab${SYNTHETIC_CONTENT_TOKEN_COUNT}-${SYNTHETIC_SAMPLING_DISTRIBUTION}${SYNTHETIC_ZIPF_ALPHA}"
echo "model: layers=${DEBUG_NUM_HIDDEN_LAYERS}, hidden=${DEBUG_HIDDEN_SIZE}, intermediate=${DEBUG_INTERMEDIATE_SIZE}"
echo "steps: ${TOTAL_TRAINING_STEPS}"

echo
echo "== 1. Attn-output-residual expert sweep =="
for router_shape in full head spectral; do
  for router_pos in "${ROUTER_POSITIONS[@]}"; do
    router_short="$(short_pos "${router_pos}")"
    if [ "${router_shape}" = "spectral" ]; then
      run_case "round5-spectral-${router_short}-exp-resid" "${router_pos}" spectral attention_output_residual full 128 4 1 false "8,32,64" "0,4,4" "1,1,1" "32,48,48"
    else
      run_case "round5-${router_shape}-${router_short}-exp-resid" "${router_pos}" "${router_shape}" attention_output_residual full 128 4 1 false
    fi
  done
done

echo
echo "== 2. Head/head sweep =="
for router_pos in "${ROUTER_POSITIONS[@]}"; do
  for expert_pos in "${EXPERT_POSITIONS[@]}"; do
    router_short="$(short_pos "${router_pos}")"
    expert_short="$(short_pos "${expert_pos}")"
    run_case "round5-headhead-r${router_short}-e${expert_short}" "${router_pos}" head "${expert_pos}" head 32 4 1 false
  done
done

echo
echo "== 3. Capacity ablation =="
for base in k layer_input; do
  base_short="$(short_pos "${base}")"
  for num_experts in 4 8 16; do
    for topk in 1 2; do
      for common in false true; do
        common_short="c0"
        if [ "${common}" = "true" ]; then
          common_short="c1"
        fi
        run_case "round5-cap-rhead-${base_short}-ne${num_experts}-topk${topk}-${common_short}" "${base}" head attention_output_residual full 128 "${num_experts}" "${topk}" "${common}"
      done
    done
  done
done

echo
echo "== 4. Spectral sample-size/stability ablation =="
for router_pos in k attention_output; do
  router_short="$(short_pos "${router_pos}")"
  for sample_size in 4096 16384 65536; do
    for spectral_warmup in 100 500; do
      for momentum in 0.0 0.9; do
        mom_short="${momentum/./p}"
        run_case "round5-spectralstab-r${router_short}-s${sample_size}-w${spectral_warmup}-m${mom_short}" "${router_pos}" spectral attention_output_residual full 128 4 1 false "8,32,64" "0,4,4" "1,1,1" "32,48,48" "${sample_size}" "${spectral_warmup}" "${momentum}"
      done
    done
  done
done

echo "Round 5 overnight experiments done: $(date)"
