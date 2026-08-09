#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT}/experiments/batched_qmse_hostmeta_20260807}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/qksieve_hostmeta_ab_32k_20260807}"
GPU="${GPU:-0}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
REPEATS="${REPEATS:-5}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"

run_case() {
  local repeat="$1"
  local name="$2"
  local hostmeta="$3"
  local output="${RUN_ROOT}/r${repeat}/${name}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  QKSIEVE_BATCH_QMSE_ALLOCATION=1 \
  QKSIEVE_BATCH_QMSE_HOST_METADATA="${hostmeta}" \
  QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1 \
  QKSIEVE_BUILD_RESIDENT_KEY_FACTORS=1 \
  QKSIEVE_RESIDENT_KEY_WORKERS=36 \
  QKSIEVE_PROFILE_INDEX_HASHES=0 \
  DRIVER="${EXPERIMENT_ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py" \
  RUN_ROOT="${output}" \
  GPU="${GPU}" \
  HISTORY_TOKENS="${HISTORY_TOKENS}" \
  EVAL_TOKENS="${EVAL_TOKENS}" \
    bash "${ROOT}/scripts/launch_qksieve_optimized_profile_20260807.sh"
}

mkdir -p "${RUN_ROOT}"
for ((repeat=1; repeat<=REPEATS; repeat++)); do
  if (( repeat % 2 )); then
    run_case "${repeat}" baseline 0
    run_case "${repeat}" hostmeta 1
  else
    run_case "${repeat}" hostmeta 1
    run_case "${repeat}" baseline 0
  fi
done

"${PYTHON}" "${EXPERIMENT_ROOT}/src/summarize_qksieve_hostmeta_ab_20260807.py" \
  "${RUN_ROOT}" | tee "${RUN_ROOT}/summary.log"
touch "${RUN_ROOT}/ALL_COMPLETE"
