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
ZIPF_ALPHAS="${ZIPF_ALPHAS:-0.7 1.0 1.3 1.6}"
RUN_PREFIX="${RUN_PREFIX:-frequency-width-skew}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000}"

SEQ_LEN="${SEQ_LEN:-128}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

for alpha in ${ZIPF_ALPHAS}; do
  alpha_tag="$(printf "%s" "${alpha}" | tr '.' 'p')"
  runs=""
  for hidden in ${HIDDEN_SIZES}; do
    if [ -n "${runs}" ]; then
      runs+=","
    fi
    runs+="${RUN_PREFIX}-zipf${alpha_tag}-h${hidden}"
  done
  output_path="../experiments/${RUN_PREFIX}-zipf${alpha_tag}-bucket-eval-step${CHECKPOINT_STEP}.json"
  echo "[$(date)] eval alpha=${alpha}: ${runs}"
  "${PYTHON_BIN}" evaluate_frequency_buckets.py \
    --config_dir ../Qwen3-0.6B \
    --checkpoint_root ../checkpoints \
    --runs "${runs}" \
    --checkpoint_step "${CHECKPOINT_STEP}" \
    --output_path "${output_path}" \
    --seq_len "${SEQ_LEN}" \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
    --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
    --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
    --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
    --synthetic_seed "${SYNTHETIC_SEED}" \
    --train_sampling_distribution zipf \
    --eval_sampling_distribution uniform \
    --synthetic_zipf_alpha "${alpha}" \
    $(if [ "${SYNTHETIC_ZIPF_SHUFFLE_RANKS}" = "true" ]; then printf "%s" "--synthetic_zipf_shuffle_ranks"; else printf "%s" "--synthetic_no_zipf_shuffle_ranks"; fi)
done
