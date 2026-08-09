#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
STRATEGY="${1:?usage: stop_tensorboard.sh STRATEGY}"
RUN_DIR="$(run_dir_for "${STRATEGY}")"
PID_FILE="${RUN_DIR}/tensorboard.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "${STRATEGY}: TensorBoard is not running"
  exit 0
fi
pid="$(cat "${PID_FILE}")"
if [[ ! "${pid}" =~ ^[0-9]+$ ]] || (( pid <= 1 )); then
  echo "refusing invalid TensorBoard PID ${pid@Q}" >&2
  exit 2
fi
cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
if [[ "${cmdline}" != *"tensorboard"* ]]; then
  echo "refusing PID ${pid}; it is not TensorBoard: ${cmdline}" >&2
  exit 3
fi
kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
rm -f "${PID_FILE}"
echo "stopped TensorBoard for ${STRATEGY}"
