#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
STRATEGY="${1:?usage: launch_strategy.sh STRATEGY}"
RUN_DIR="$(run_dir_for "${STRATEGY}")"
mkdir -p "${RUN_DIR}"
PID_FILE="${RUN_DIR}/controller.pid"
if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "${STRATEGY} is already running with PID ${OLD_PID}" >&2
    exit 2
  fi
  rm -f "${PID_FILE}"
fi
if [[ "${FOREGROUND:-0}" == "1" ]]; then
  exec bash "${PROJECT_ROOT}/scripts/run_strategy_foreground.sh" "${STRATEGY}"
fi
nohup setsid bash "${PROJECT_ROOT}/scripts/run_strategy_foreground.sh" "${STRATEGY}" \
  >> "${RUN_DIR}/nohup.log" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"
sleep 1
if ! kill -0 "${PID}" 2>/dev/null; then
  echo "launcher exited early; inspect ${RUN_DIR}/nohup.log" >&2
  exit 1
fi
echo "started strategy=${STRATEGY} pid=${PID} gpus=${GPU_LIST}"
echo "log: ${RUN_DIR}/nohup.log"
