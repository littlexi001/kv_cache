#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs}"
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
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/workspace/Qwen3-0.6B}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_DIR}/output/eval_data}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/output/downstream_eval_results}"
RUN_NAME="${RUN_NAME:-$(basename "${LATEST_RUN}")}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
MC_LIMIT="${MC_LIMIT:-200}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-2048}"

TASK_ARGS=()
for path in "${EVAL_DATA_DIR}"/*.jsonl; do
  name="$(basename "${path}" .jsonl)"
  if [[ "${name}" == "manifest" ]]; then
    continue
  fi
  TASK_ARGS+=(--mc_task "${name}=${path}")
done

if [[ "${#TASK_ARGS[@]}" -eq 0 ]]; then
  echo "No JSONL eval tasks found in ${EVAL_DATA_DIR}."
  echo "Run: bash scripts/prepare_downstream_eval_data.sh"
  exit 1
fi

python "${PROJECT_DIR}/eval/eval_checkpoint_vs_baseline.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --baseline_model_path "${BASELINE_MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --output_dir "${RESULT_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --mc_limit "${MC_LIMIT}" \
  --eval_seq_len "${EVAL_SEQ_LEN}" \
  "${TASK_ARGS[@]}" \
  "$@"

echo "wrote eval results: ${RESULT_DIR}"
