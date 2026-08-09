#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_meantail_alpha05_generality_128k_6gpu}"
WORKER="${ROOT}/scripts/run_qksieve_valuesketch_alpha_pair_20260801.sh"
VARIANT="${VARIANT:-qksieve_keymse_requestlocal_meantail_k1280_c32}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

GPU_IDS="0,1" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="sports_both:20260831 computer:20260833 religion:20260835" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/same_remaining_worker.log" 2>&1 &
pid0=$!

GPU_IDS="2,3" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="sports_both:20260931 medicine:20260932 computer:20260933" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/holdout_a_worker.log" 2>&1 &
pid1=$!

GPU_IDS="4,5" ALPHA="0.5" VARIANT="${VARIANT}" RUN_ROOT="${RUN_ROOT}" \
TOPICS="politics:20260934 religion:20260935 mixed_b:20260936" \
  bash "${WORKER}" >"${RUN_ROOT}/logs/holdout_b_worker.log" 2>&1 &
pid2=$!

status=0
for pid in "${pid0}" "${pid1}" "${pid2}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
