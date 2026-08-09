#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STRATEGY="${1:?usage: run_strategy_foreground.sh STRATEGY}"
if [[ "${STRATEGY}" == "base_eval" ]]; then
  STRATEGY_FILE="$(strategy_path native_rope)"
  EXTRA=(--eval-only)
else
  STRATEGY_FILE="$(strategy_path "${STRATEGY}")"
  EXTRA=()
fi
RUN_DIR="$(run_dir_for "${STRATEGY}")"
mkdir -p "${RUN_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
echo "$$" > "${RUN_DIR}/controller.pid"
cleanup() { rm -f "${RUN_DIR}/controller.pid"; }
trap cleanup EXIT

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/validate_contract.py" \
  --model-root "${MODEL_ROOT}" \
  --dclm-root "${DCLM_ROOT}" \
  --strategy "${STRATEGY_FILE}" \
  --sequence-length "${SEQ_LEN}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/orchestrate.py" \
  --model-root "${MODEL_ROOT}" \
  --dclm-root "${DCLM_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --strategy "${STRATEGY_FILE}" \
  --native-strategy "$(strategy_path native_rope)" \
  --sequence-length "${SEQ_LEN}" \
  --micro-batch "${MICRO_BATCH}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --target-tokens "${TARGET_TOKENS}" \
  --milestone-tokens "${MILESTONE_TOKENS}" \
  --total-steps "${TOTAL_STEPS}" \
  --milestones "${MILESTONES}" \
  --learning-rate "${LEARNING_RATE}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --adam-beta1 "${ADAM_BETA1}" \
  --adam-beta2 "${ADAM_BETA2}" \
  --adam-epsilon "${ADAM_EPSILON}" \
  --seed "${SEED}" \
  --data-seed "${DATA_SEED}" \
  --num-workers "${NUM_WORKERS}" \
  --train-files "${TRAIN_FILES}" \
  --validation-files "${VALIDATION_FILES}" \
  --eval-lengths "${EVAL_LENGTHS}" \
  --ruler-samples-per-task "${RULER_SAMPLES_PER_TASK}" \
  --ppl-blocks "${PPL_BLOCKS}" \
  --run-longbench "${RUN_LONGBENCH}" \
  --longbench-tasks "${LONGBENCH_TASKS}" \
  --longbench-samples-per-task "${LONGBENCH_SAMPLES_PER_TASK}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --dtype "${DTYPE}" \
  --attention-implementation "${ATTN_IMPLEMENTATION}" \
  --initialization "${INITIALIZATION}" \
  --logging-steps "${LOGGING_STEPS}" \
  --tensorboard "${TENSORBOARD_ENABLED}" \
  "${EXTRA[@]}"
