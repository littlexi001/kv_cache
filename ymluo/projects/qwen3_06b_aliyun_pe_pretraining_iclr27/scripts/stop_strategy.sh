#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
STRATEGY="${1:?usage: stop_strategy.sh STRATEGY}"
RUN_DIR="$(run_dir_for "${STRATEGY}")"
PID_FILE="${RUN_DIR}/controller.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "${STRATEGY}: no PID file; already stopped"
  exit 0
fi
PID="$(cat "${PID_FILE}")"
if [[ ! "${PID}" =~ ^[0-9]+$ ]] || (( PID <= 1 )); then
  echo "refusing invalid PID ${PID@Q}" >&2
  exit 2
fi
if ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "${STRATEGY}: stale PID file removed"
  exit 0
fi
CMDLINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null || true)"
if [[ "${CMDLINE}" != *"qwen3_06b_aliyun_pe_pretraining_iclr27"* ]]; then
  echo "refusing to stop PID ${PID}; command does not belong to this package: ${CMDLINE}" >&2
  exit 3
fi
kill -TERM -- "-${PID}"
for _ in $(seq 1 30); do
  kill -0 "${PID}" 2>/dev/null || break
  sleep 1
done
if kill -0 "${PID}" 2>/dev/null; then
  echo "TERM timeout; sending KILL only to process group ${PID}" >&2
  kill -KILL -- "-${PID}"
fi
rm -f "${PID_FILE}"
echo "stopped ${STRATEGY}"

