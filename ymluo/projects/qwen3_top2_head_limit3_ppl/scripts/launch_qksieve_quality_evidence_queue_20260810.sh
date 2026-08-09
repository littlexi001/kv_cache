#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
WAIT_FOR="${WAIT_FOR:-${ROOT}/results/20260810_qksieve_persistent_kv_v2}"
QUEUE_ROOT="${QUEUE_ROOT:-${ROOT}/results/20260810_qksieve_quality_evidence_queue_v2}"
RULER="${ROOT}/scripts/launch_qksieve_robust_ruler_20260810.sh"
MULTIMODEL="${ROOT}/scripts/launch_qksieve_robust_multimodel_longbench_20260810.sh"
FULL_LONGBENCH="${ROOT}/scripts/launch_qksieve_robust_llama_full_longbench_20260810.sh"

mkdir -p "${QUEUE_ROOT}/logs"
touch "${QUEUE_ROOT}/RUNNING"
rm -f "${QUEUE_ROOT}/ALL_COMPLETE" "${QUEUE_ROOT}/FAILED"

while [[ -f "${WAIT_FOR}/RUNNING" ]]; do sleep 30; done
if [[ ! -f "${WAIT_FOR}/ALL_COMPLETE" ]]; then
  echo "upstream persistent-KV v2 evidence did not complete" \
    >"${QUEUE_ROOT}/logs/upstream_failure.log"
  touch "${QUEUE_ROOT}/FAILED"
  rm -f "${QUEUE_ROOT}/RUNNING"
  exit 1
fi

ROOT="${ROOT}" bash "${RULER}" \
  >"${QUEUE_ROOT}/logs/ruler.log" 2>&1 || {
    touch "${QUEUE_ROOT}/FAILED"
    rm -f "${QUEUE_ROOT}/RUNNING"
    exit 1
  }
ROOT="${ROOT}" bash "${MULTIMODEL}" \
  >"${QUEUE_ROOT}/logs/multimodel.log" 2>&1 || {
    touch "${QUEUE_ROOT}/FAILED"
    rm -f "${QUEUE_ROOT}/RUNNING"
    exit 1
  }
ROOT="${ROOT}" bash "${FULL_LONGBENCH}" \
  >"${QUEUE_ROOT}/logs/full_longbench.log" 2>&1 || {
    touch "${QUEUE_ROOT}/FAILED"
    rm -f "${QUEUE_ROOT}/RUNNING"
    exit 1
  }

rm -f "${QUEUE_ROOT}/RUNNING" "${QUEUE_ROOT}/FAILED"
touch "${QUEUE_ROOT}/ALL_COMPLETE"
