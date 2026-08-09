#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_tilttail16_alpha05_weak3_6gpu}"
WORKER="${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh"
VARIANT="qksieve_keymse_requestlocal_tilttail16_k1280_c32"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

GPU_IDS="0,1" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="medicine:20260832" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/medicine_worker.log" 2>&1 &
pid0=$!

GPU_IDS="2,3" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="mixed_b:20260836" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/mixed_worker.log" 2>&1 &
pid1=$!

GPU_IDS="4,5" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="politics:20260834" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/politics_worker.log" 2>&1 &
pid2=$!

status=0
for pid in "${pid0}" "${pid1}" "${pid2}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
