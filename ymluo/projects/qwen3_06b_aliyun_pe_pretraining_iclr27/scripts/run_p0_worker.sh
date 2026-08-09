#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "usage: bash scripts/run_p0_worker.sh TASK_ID  # TASK_ID=0..3" >&2
  exit 2
fi
: "${GPU_LIST:=0,1,2,3,4,5,6,7}"
export GPU_LIST
source "${SCRIPT_DIR}/common.sh"

VALIDATE_EXTRA=()
if [[ "${ALLOW_NON_100B:-0}" == "1" ]]; then
  VALIDATE_EXTRA+=(--allow-target-token-override)
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/validate_pretrain_protocol.py" \
  --gpu-list "${GPU_LIST}" \
  --initialization "${INITIALIZATION}" \
  --sequence-length "${SEQ_LEN}" \
  --micro-batch "${MICRO_BATCH}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --target-tokens "${TARGET_TOKENS}" \
  --learning-rate "${LEARNING_RATE}" \
  "${VALIDATE_EXTRA[@]}"

STRATEGY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/p0_four_machine_plan.py" --machine-id "${TASK_ID}" --field strategy)"
ROLE="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/p0_four_machine_plan.py" --machine-id "${TASK_ID}" --field role)"
if [[ "${STRATEGY}" == optimized_phase_complementary* ]] && \
   [[ ! -f "$(strategy_path "${STRATEGY}")" ]]; then
  bash "${SCRIPT_DIR}/prepare_p0_strategies.sh"
fi
"${PYTHON_BIN}" "${PROJECT_ROOT}/src/validate_contract.py" \
  --model-root "${MODEL_ROOT}" \
  --dclm-root "${DCLM_ROOT}" \
  --strategy "$(strategy_path "${STRATEGY}")" \
  --sequence-length "${SEQ_LEN}"

echo "p0_task=${TASK_ID} strategy=${STRATEGY} role=${ROLE}"
echo "gpus=${GPU_LIST} global_batch=${GLOBAL_BATCH_SIZE} target_tokens=${TARGET_TOKENS}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=complete; no process was started"
  exit 0
fi
bash "${SCRIPT_DIR}/launch_strategy.sh" "${STRATEGY}"
bash "${SCRIPT_DIR}/start_tensorboard.sh" "${STRATEGY}" || \
  echo "warning: training is running but TensorBoard did not start" >&2
