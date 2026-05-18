#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME_VALUE="${RUN_NAME:-unet8-random-quad-lm-ddp}"
CUDA_DEVICES_VALUE="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
SAFE_CUDA_DEVICES="${CUDA_DEVICES_VALUE//,/}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/ddp_${RUN_NAME_VALUE}_cuda${SAFE_CUDA_DEVICES}_${TIMESTAMP}.log}"
PID_PATH="${PID_PATH:-${LOG_DIR}/ddp_${RUN_NAME_VALUE}.pid}"

CUDA_DEVICES="${CUDA_DEVICES_VALUE}" nohup bash "${SCRIPT_DIR}/run_ddp_train.sh" "$@" >"${LOG_PATH}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_PATH}"

echo "started random quadruple DDP training"
echo "cuda_devices: ${CUDA_DEVICES_VALUE}"
echo "pid: ${PID}"
echo "log: ${LOG_PATH}"
echo "pid_file: ${PID_PATH}"
