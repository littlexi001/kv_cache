#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_ID="${1:-${MACHINE_ID:-}}"
if [[ -z "${MACHINE_ID}" ]]; then
  echo "usage: bash scripts/run_four_machine_worker.sh MACHINE_ID" >&2
  echo "MACHINE_ID must be one of 0, 1, 2, or 3" >&2
  exit 2
fi

# The four-machine protocol assumes one eight-GPU interactive instance per
# condition. A caller can still override this before launch when necessary.
: "${GPU_LIST:=0,1,2,3,4,5,6,7}"
export GPU_LIST
source "${SCRIPT_DIR}/common.sh"

STRATEGY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/four_machine_plan.py" \
  --machine-id "${MACHINE_ID}" --field strategy)"
ROLE="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/four_machine_plan.py" \
  --machine-id "${MACHINE_ID}" --field role)"

echo "machine_id=${MACHINE_ID}"
echo "strategy=${STRATEGY}"
echo "role=${ROLE}"
echo "gpus=${GPU_LIST}"
exec bash "${SCRIPT_DIR}/launch_strategy.sh" "${STRATEGY}"
