#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "usage: bash scripts/run_pretrain_worker.sh TASK_ID  # TASK_ID=0..15" >&2
  exit 2
fi
: "${GPU_LIST:=0,1,2,3,4,5,6,7}"
export GPU_LIST
source "${SCRIPT_DIR}/common.sh"
"${PYTHON_BIN}" "${PROJECT_ROOT}/src/validate_pretrain_protocol.py" \
  --gpu-list "${GPU_LIST}" \
  --initialization "${INITIALIZATION}" \
  --sequence-length "${SEQ_LEN}" \
  --micro-batch "${MICRO_BATCH}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --target-tokens "${TARGET_TOKENS}" \
  --learning-rate "${LEARNING_RATE}"
STRATEGY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/sixteen_machine_plan.py" --machine-id "${TASK_ID}" --field strategy)"
FAMILY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/sixteen_machine_plan.py" --machine-id "${TASK_ID}" --field family)"
METHOD_DOC="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/sixteen_machine_plan.py" --machine-id "${TASK_ID}" --field method_doc)"
echo "task_id=${TASK_ID} strategy=${STRATEGY} family=${FAMILY}"
echo "method_doc=${PROJECT_ROOT}/${METHOD_DOC}"
echo "gpus=${GPU_LIST} global_batch=${GLOBAL_BATCH_SIZE} target_tokens=${TARGET_TOKENS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=complete; no process was started"
  exit 0
fi
bash "${SCRIPT_DIR}/launch_strategy.sh" "${STRATEGY}"
bash "${SCRIPT_DIR}/start_tensorboard.sh" "${STRATEGY}" || \
  echo "warning: training is running but TensorBoard did not start" >&2
