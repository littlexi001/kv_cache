#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME_VALUE="${RUN_NAME:-unet8-small-intervals123-lm}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/train_${RUN_NAME_VALUE}_${TIMESTAMP}.log}"
PID_PATH="${PID_PATH:-${LOG_DIR}/train_${RUN_NAME_VALUE}.pid}"

nohup bash "${SCRIPT_DIR}/run_train.sh" "$@" >"${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_PATH}"

echo "started interval subseq training"
echo "pid: ${PID}"
echo "log: ${LOG_PATH}"
echo "pid_file: ${PID_PATH}"
