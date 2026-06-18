#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

RUN_NAME="${RUN_NAME:-routed_top4_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

export RUN_NAME

nohup bash "${PROJECT_DIR}/scripts/train_8x80g.sh" "$@" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "started routed top4 training"
echo "pid: ${PID}"
echo "run_name: ${RUN_NAME}"
echo "log: ${LOG_FILE}"
echo "output root: ${OUTPUT_ROOT:-/mnt/workspace/routed_top4_qwen3_0p6b_runs}"
