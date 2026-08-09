#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Yarn-Llama-2-7b-128k}"
CASE_RUNNER="${ROOT}/scripts/run_qksieve_persistent_kv_case_20260810.sh"
SUMMARY_RUNNER="${ROOT}/src/summarize_qksieve_persistent_kv_20260810.py"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_persistent_kv_v2}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}/logs"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

status=0
run_pair() {
  local devices="$1" length="$2" method
  for method in full qksieve_robust; do
    CUDA_VISIBLE_DEVICES="${devices}" \
    GPU_TAG="${devices//,/-}" \
    ROOT="${ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    MODEL="${MODEL}" \
    HISTORY_TOKENS="${length}" \
    METHOD="${method}" \
    BRANCH_COUNT=4 \
    BRANCH_STEPS=32 \
    APPEND_STEPS=128 \
    bash "${CASE_RUNNER}" || return 1
  done
}

# Run lengths sequentially so first-request timings are not contaminated by
# concurrent CUDA extension loading or host-side eigensolver contention.
run_pair 0,1 32768 || status=1
if [[ ${status} -eq 0 ]]; then
  run_pair 0,1,2 65536 || status=1
fi

if [[ ${status} -eq 0 ]]; then
  "${PYTHON}" "${SUMMARY_RUNNER}" \
    --run_root "${RUN_ROOT}" \
    --output "${RUN_ROOT}/summary.json" \
    >"${RUN_ROOT}/logs/summary.log" 2>&1 || status=1
fi

rm -f "${RUN_ROOT}/RUNNING"
if [[ ${status} -eq 0 ]]; then
  touch "${RUN_ROOT}/ALL_COMPLETE"
else
  touch "${RUN_ROOT}/FAILED"
fi
exit "${status}"
