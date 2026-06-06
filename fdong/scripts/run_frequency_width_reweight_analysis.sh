#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

REPO_ROOT="$(cd ../.. && pwd)"
DEFAULT_VENV_PYTHON="${REPO_ROOT}/.venv-transformers451/bin/python3"
if [ -x "${DEFAULT_VENV_PYTHON}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_VENV_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

HIDDEN_SIZES="${HIDDEN_SIZES:-64 96}"
RUN_PREFIX="${RUN_PREFIX:-frequency-width-reweight}"
FREQUENCY_LOSS_WEIGHTING="${FREQUENCY_LOSS_WEIGHTING:-inverse_sqrt}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
MODES="${MODES:-learning_curve,gradients,svd,probe,lm_bias}"
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/${RUN_PREFIX}-${FREQUENCY_LOSS_WEIGHTING}-analysis.json}"

SEQ_LEN="${SEQ_LEN:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SVD_SAMPLES="${SVD_SAMPLES:-256}"
GRADIENT_SAMPLES="${GRADIENT_SAMPLES:-128}"
GRADIENT_BATCH_SIZE="${GRADIENT_BATCH_SIZE:-16}"

SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.3}"

run_specs=""
for hidden in ${HIDDEN_SIZES}; do
  if [ -n "${run_specs}" ]; then
    run_specs+=","
  fi
  run_specs+="reweight_h${hidden}:${RUN_PREFIX}-zipf-${FREQUENCY_LOSS_WEIGHTING}-h${hidden}:zipf"
done

"${PYTHON_BIN}" analyze_frequency_width_dynamics.py \
  --config_dir ../Qwen3-0.6B \
  --checkpoint_root ../checkpoints \
  --run_specs "${run_specs}" \
  --output_path "${OUTPUT_PATH}" \
  --modes "${MODES}" \
  --checkpoint_steps "${CHECKPOINT_STEPS}" \
  --seq_len "${SEQ_LEN}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --svd_samples "${SVD_SAMPLES}" \
  --gradient_samples "${GRADIENT_SAMPLES}" \
  --gradient_batch_size "${GRADIENT_BATCH_SIZE}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
  --synthetic_seed "${SYNTHETIC_SEED}" \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}"
