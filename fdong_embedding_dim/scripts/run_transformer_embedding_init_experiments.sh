#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FDONG_SCRIPTS="${REPO_ROOT}/fdong/scripts"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_PREFIX="${RUN_PREFIX:-embedding-dim-transformer}"
INIT_ROOT="${INIT_ROOT:-${REPO_ROOT}/fdong_embedding_dim/outputs/transformer_init_checkpoints}"
ANALYSIS_OUT="${ANALYSIS_OUT:-${REPO_ROOT}/fdong_embedding_dim/outputs/transformer_spectral_occupation.json}"

HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-256}"
HEAD_DIM="${HEAD_DIM:-32}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
SEQ_LEN="${SEQ_LEN:-128}"
SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.3}"
TRAINING_SEED="${TRAINING_SEED:-20260608}"
RUN_FILTER="${RUN_FILTER:-}"

mkdir -p "${INIT_ROOT}" "${REPO_ROOT}/fdong/logs" "${REPO_ROOT}/fdong/checkpoints"

run_train_one() {
  local init_mode="$1"
  local run_name="${RUN_PREFIX}-${init_mode}-h${HIDDEN_SIZE}"
  local init_path="${INIT_ROOT}/${run_name}.pth"
  local init_meta="${INIT_ROOT}/${run_name}.json"
  local log_path="${REPO_ROOT}/fdong/logs/${run_name}.train.log"

  if [[ -n "${RUN_FILTER}" && "${run_name}" != *"${RUN_FILTER}"* ]]; then
    return 0
  fi

  echo "================================================================================"
  echo "Create init checkpoint ${run_name}"
  "${PYTHON_BIN}" -u "${REPO_ROOT}/fdong_embedding_dim/scripts/create_transformer_embedding_init_checkpoint.py" \
    --config_dir "${REPO_ROOT}/fdong/Qwen3-0.6B" \
    --output_path "${init_path}" \
    --runtime_config_path "${init_meta}" \
    --init_mode "${init_mode}" \
    --debug_vocab_size "$((SYNTHETIC_CONTENT_TOKEN_COUNT + 1))" \
    --debug_hidden_size "${HIDDEN_SIZE}" \
    --debug_intermediate_size "${INTERMEDIATE_SIZE}" \
    --debug_num_hidden_layers 2 \
    --debug_num_attention_heads 4 \
    --debug_num_key_value_heads 2 \
    --debug_head_dim "${HEAD_DIM}" \
    --debug_max_position_embeddings 256 \
    --seq_len "${SEQ_LEN}" \
    --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
    --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
    --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}"

  echo "Train ${run_name}"
  (
    cd "${FDONG_SCRIPTS}"
    env \
      DATASET_TYPE="hierarchical_pattern" \
      SYNTHETIC_SAMPLING_DISTRIBUTION="zipf" \
      SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA}" \
      TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      LOCAL_BATCH_SIZE="16" \
      GLOBAL_BATCH_SIZE="16" \
      SEQ_LEN="${SEQ_LEN}" \
      SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES}" \
      SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
      SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
      DEBUG_VOCAB_SIZE="$((SYNTHETIC_CONTENT_TOKEN_COUNT + 1))" \
      DEBUG_HIDDEN_SIZE="${HIDDEN_SIZE}" \
      DEBUG_INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE}" \
      DEBUG_NUM_HIDDEN_LAYERS="2" \
      DEBUG_NUM_ATTENTION_HEADS="4" \
      DEBUG_NUM_KEY_VALUE_HEADS="2" \
      DEBUG_HEAD_DIM="${HEAD_DIM}" \
      DEBUG_MAX_POSITION_EMBEDDINGS="256" \
      USE_MOE="false" \
      LR="1e-3" \
      WARMUP_STEPS="100" \
      TRAINING_SEED="${TRAINING_SEED}" \
      INIT_CHECKPOINT="${init_path}" \
      CKPT_DIR="../checkpoints/${run_name}" \
      bash run_inverse_kv_local_experiment.sh > "${log_path}" 2>&1
  )
  echo "Done train ${run_name}; log=${log_path}"
}

# 2.1 / 2.4 controlled embedding initialization settings.
run_train_one "spread"
run_train_one "packed_common"
run_train_one "packed_negative_common"

run_specs=""
for init_mode in spread packed_common packed_negative_common; do
  run_name="${RUN_PREFIX}-${init_mode}-h${HIDDEN_SIZE}"
  if [[ -n "${RUN_FILTER}" && "${run_name}" != *"${RUN_FILTER}"* ]]; then
    continue
  fi
  if [[ -n "${run_specs}" ]]; then
    run_specs+=",";
  fi
  run_specs+="${init_mode}:${run_name}:zipf"
done

if [[ -n "${run_specs}" ]]; then
  echo "================================================================================"
  echo "Analyze spectral occupation"
  "${PYTHON_BIN}" -u "${REPO_ROOT}/fdong_embedding_dim/scripts/analyze_transformer_spectral_occupation.py" \
    --config_dir "${REPO_ROOT}/fdong/Qwen3-0.6B" \
    --checkpoint_root "${REPO_ROOT}/fdong/checkpoints" \
    --run_specs "${run_specs}" \
    --checkpoint_steps "all" \
    --output_path "${ANALYSIS_OUT}" \
    --seq_len "${SEQ_LEN}" \
    --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
    --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
    --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}"
fi

echo "================================================================================"
echo "Transformer embedding-init experiments finished."
