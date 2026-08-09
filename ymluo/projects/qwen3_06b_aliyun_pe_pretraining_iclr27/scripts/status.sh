#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
mapfile -t strategies < <(
  { echo base_eval; find "${PROJECT_ROOT}/configs/strategies" -maxdepth 1 -type f -name '*.json' \
      -printf '%f\n' | sed 's/\.json$//' | sort; } | awk '!seen[$0]++'
)
for strategy in "${strategies[@]}"; do
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
  printf '%-28s %s\n' "${strategy}" "${state}"
  if [[ -f "${run_dir}/controller_events.jsonl" ]]; then
    tail -n 1 "${run_dir}/controller_events.jsonl" | sed 's/^/  last_event: /'
  fi
done
