#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

RUN_NAME="${RUN_NAME:-qwen15-moe-0p6b-real-attn-cluster}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

nohup bash "${SCRIPT_DIR}/run_8gpu.sh" > "${LOG_FILE}" 2>&1 &
echo "$!" > "${PID_FILE}"

echo "started ${RUN_NAME}"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "resume: RESUME_FROM_CHECKPOINT=auto bash ${SCRIPT_DIR}/nohup_train.sh"
echo "tensorboard: tensorboard --logdir ${PROJECT_DIR}/outputs --host 0.0.0.0 --port 6006"
