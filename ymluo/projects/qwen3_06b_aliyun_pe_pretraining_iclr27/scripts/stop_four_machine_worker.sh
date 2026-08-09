#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_ID="${1:-${MACHINE_ID:-}}"
if [[ -z "${MACHINE_ID}" ]]; then
  echo "usage: bash scripts/stop_four_machine_worker.sh MACHINE_ID" >&2
  exit 2
fi
source "${SCRIPT_DIR}/common.sh"
STRATEGY="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/four_machine_plan.py" \
  --machine-id "${MACHINE_ID}" --field strategy)"
exec bash "${SCRIPT_DIR}/stop_strategy.sh" "${STRATEGY}"
