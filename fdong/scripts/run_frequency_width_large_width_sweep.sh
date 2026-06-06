#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints ../experiments

REPO_ROOT="$(cd ../.. && pwd)"
DEFAULT_VENV_PYTHON="${REPO_ROOT}/.venv-transformers451/bin/python3"
if [ -x "${DEFAULT_VENV_PYTHON}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_VENV_PYTHON}}"
  export PATH="${REPO_ROOT}/.venv-transformers451/bin:${PATH}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Main question: as width grows beyond h96, does Zipf tail eventually approach
# head performance, or does a persistent tail gap remain under fixed token budget?
HIDDEN_SIZES="${HIDDEN_SIZES:-128 256 512 1024}"
RUN_PREFIX="${RUN_PREFIX:-frequency-width-large}"
CONDITION="${CONDITION:-zipf}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
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
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.3}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
ANALYSIS_NUM_SAMPLES="${ANALYSIS_NUM_SAMPLES:-512}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-16}"
ANALYSIS_MODES="${ANALYSIS_MODES:-learning_curve,lm_bias,svd,probe}"

runs_csv() {
  local result=""
  for hidden in ${HIDDEN_SIZES}; do
    if [ -n "${result}" ]; then
      result+=","
    fi
    result+="${RUN_PREFIX}-${CONDITION}-h${hidden}"
  done
  printf "%s" "${result}"
}

run_specs_csv() {
  local result=""
  for hidden in ${HIDDEN_SIZES}; do
    if [ -n "${result}" ]; then
      result+=","
    fi
    result+="h${hidden}:${RUN_PREFIX}-${CONDITION}-h${hidden}:${CONDITION}"
  done
  printf "%s" "${result}"
}

train_one() {
  local hidden="$1"
  local intermediate=$((hidden * 2))
  local head_dim=$((hidden / 4))
  local run_name="${RUN_PREFIX}-${CONDITION}-h${hidden}"
  local log_path="../logs/${run_name}.train.log"

  echo "[$(date)] train ${run_name}"
  env \
    DATASET_TYPE="hierarchical_pattern" \
    SYNTHETIC_SAMPLING_DISTRIBUTION="${CONDITION}" \
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

echo "========== frequency-width large-width sweep =========="
echo "condition=${CONDITION}, zipf_alpha=${SYNTHETIC_ZIPF_ALPHA}"
echo "hidden_sizes=${HIDDEN_SIZES}"
echo "steps=${TOTAL_TRAINING_STEPS}, save_interval=${SAVE_INTERVAL}"
echo "======================================================"

for hidden in ${HIDDEN_SIZES}; do
  train_one "${hidden}"
done

final_eval_path="../experiments/${RUN_PREFIX}-${CONDITION}-bucket-eval-step${TOTAL_TRAINING_STEPS}.json"
echo "[$(date)] final bucket eval -> ${final_eval_path}"
"${PYTHON_BIN}" evaluate_frequency_buckets.py \
  --config_dir ../Qwen3-0.6B \
  --checkpoint_root ../checkpoints \
  --runs "$(runs_csv)" \
  --checkpoint_step "${TOTAL_TRAINING_STEPS}" \
  --output_path "${final_eval_path}" \
  --seq_len "${SEQ_LEN}" \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
  --synthetic_seed "${SYNTHETIC_SEED}" \
  --train_sampling_distribution "${CONDITION}" \
  --eval_sampling_distribution uniform \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}" \
  $(if [ "${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" = "true" ]; then printf "%s" "--synthetic_zipf_shuffle_ranks"; else printf "%s" "--synthetic_no_zipf_shuffle_ranks"; fi)

analysis_path="../experiments/${RUN_PREFIX}-${CONDITION}-analysis.json"
echo "[$(date)] checkpoint analysis -> ${analysis_path}"
"${PYTHON_BIN}" analyze_frequency_width_dynamics.py \
  --config_dir ../Qwen3-0.6B \
  --checkpoint_root ../checkpoints \
  --run_specs "$(run_specs_csv)" \
  --output_path "${analysis_path}" \
  --modes "${ANALYSIS_MODES}" \
  --checkpoint_steps all \
  --seq_len "${SEQ_LEN}" \
  --num_samples "${ANALYSIS_NUM_SAMPLES}" \
  --batch_size "${ANALYSIS_BATCH_SIZE}" \
  --svd_samples "${ANALYSIS_NUM_SAMPLES}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
  --synthetic_seed "${SYNTHETIC_SEED}" \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}" \
  $(if [ "${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" = "true" ]; then printf "%s" "--synthetic_zipf_shuffle_ranks"; else printf "%s" "--synthetic_no_zipf_shuffle_ranks"; fi)

echo "[$(date)] done"
echo "final eval: ${final_eval_path}"
echo "analysis: ${analysis_path}"
