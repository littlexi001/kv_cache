#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-${TASK_ID:-}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "usage: bash scripts/stop_pretrain_worker.sh TASK_ID" >&2
  exit 2
fi
source "${SCRIPT_DIR}/common.sh"
STRATEGY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/sixteen_machine_plan.py" --machine-id "${TASK_ID}" --field strategy)"
bash "${SCRIPT_DIR}/stop_strategy.sh" "${STRATEGY}"
bash "${SCRIPT_DIR}/stop_tensorboard.sh" "${STRATEGY}" || true
