#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs}"
HELDOUT_DATASET="${HELDOUT_DATASET:-wikitext103_validation}"
HELDOUT_TEXT_DIR="${HELDOUT_TEXT_DIR:-${PROJECT_DIR}/output/heldout_text}"
HELDOUT_TEXT_PATH="${HELDOUT_TEXT_PATH:-${HELDOUT_TEXT_DIR}/${HELDOUT_DATASET}.txt}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/workspace/Qwen3-0.6B}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/output/heldout_ppl_results}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"
EVAL_TEXT_MAX_CHARS="${EVAL_TEXT_MAX_CHARS:-5000000}"
EVAL_TEXT_MAX_BATCHES="${EVAL_TEXT_MAX_BATCHES:-0}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  LATEST_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'routed_top4_*' | sort | tail -n 1)"
  if [[ -z "${LATEST_RUN}" ]]; then
    echo "No routed_top4_* run found under ${OUTPUT_ROOT}."
    echo "Set CHECKPOINT_DIR=/path/to/checkpoint or OUTPUT_ROOT=/path/to/runs."
    exit 1
  fi
  CHECKPOINT_DIR="${LATEST_RUN}/latest_checkpoint"
else
  LATEST_RUN="$(dirname "${CHECKPOINT_DIR}")"
fi

if [[ ! -f "${HELDOUT_TEXT_PATH}" ]]; then
  echo "Held-out text file not found: ${HELDOUT_TEXT_PATH}"
  echo "Run: bash scripts/prepare_heldout_ppl_text.sh"
  exit 1
fi

RUN_NAME="${RUN_NAME:-$(basename "${LATEST_RUN}")}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/${RUN_NAME}_${HELDOUT_DATASET}_$(date +%Y%m%d_%H%M%S)}"

python "${PROJECT_DIR}/eval/eval_checkpoint_vs_baseline.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --baseline_model_path "${BASELINE_MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --output_dir "${RESULT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --eval_seq_len "${EVAL_SEQ_LEN}" \
  --eval_text_path "${HELDOUT_TEXT_PATH}" \
  --eval_text_max_chars "${EVAL_TEXT_MAX_CHARS}" \
  --eval_text_max_batches "${EVAL_TEXT_MAX_BATCHES}" \
  "$@"

python "${PROJECT_DIR}/eval/summarize_eval_results.py" "${RESULT_DIR}/summary.json"
echo "wrote held-out PPL results: ${RESULT_DIR}"
