#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
STRATEGY="${1:?usage: start_tensorboard.sh STRATEGY}"
if [[ "${TENSORBOARD_ENABLED}" != "1" ]]; then
  echo "TensorBoard disabled by TENSORBOARD_ENABLED=${TENSORBOARD_ENABLED}"
  exit 0
fi
RUN_DIR="$(run_dir_for "${STRATEGY}")"
mkdir -p "${RUN_DIR}/tensorboard"
PID_FILE="${RUN_DIR}/tensorboard.pid"
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "TensorBoard already running pid=${pid} port=${TENSORBOARD_PORT}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi
nohup setsid "${PYTHON_BIN}" -m tensorboard.main \
  --logdir "${RUN_DIR}/tensorboard" \
  --host "${TENSORBOARD_HOST}" \
  --port "${TENSORBOARD_PORT}" \
  > "${RUN_DIR}/tensorboard.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
sleep 1
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "TensorBoard failed; inspect ${RUN_DIR}/tensorboard.log" >&2
  exit 1
fi
echo "TensorBoard started pid=${pid} http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
