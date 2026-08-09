#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
printf '%-10s %-28s %s\n' "MACHINE" "STRATEGY" "LOCAL STATUS"
for machine_id in 0 1 2 3; do
  strategy="$(${PYTHON_BIN} "${PROJECT_ROOT}/src/four_machine_plan.py" \
    --machine-id "${machine_id}" --field strategy)"
  run_dir="$(run_dir_for "${strategy}")"
  pid_file="${run_dir}/controller.pid"
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      state="RUNNING pid=${pid}"
    else
      state="STALE_PID"
    fi
  elif [[ -f "${run_dir}/controller.done" ]]; then
    state="COMPLETE"
  else
    state="NOT_STARTED"
  fi
  printf '%-10s %-28s %s\n' "${machine_id}" "${strategy}" "${state}"
done
