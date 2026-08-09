#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
SCRIPT="${ROOT}/scripts/run_qksieve_native128k_valuesketch_topics_20260801.sh"
VARIANT="qksieve_keymse_requestlocal_sampled_k1280_c32"

recent128="${ROOT}/results/20260801_qksieve_recent128_native128k_targeted_v1"
recent256="${ROOT}/results/20260801_qksieve_recent256_native128k_targeted_v1"
ordinary="${ROOT}/results/20260801_qksieve_keyonly_native128k_same6_v25_6gpu"
mkdir -p "${recent128}" "${recent256}" "${ordinary}/launcher_logs"

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${recent128}" \
  GPU_IDS="0,1" \
  TOPICS="sports_both:20260831 medicine:20260832 computer:20260833" \
  VARIANTS="${VARIANT}" \
  PROTECT_RECENT_TOKENS=128 \
  bash "${SCRIPT}" >"${recent128}/launcher.log" 2>&1 &
pid128=$!

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${recent256}" \
  GPU_IDS="2,3" \
  TOPICS="sports_both:20260831 medicine:20260832 computer:20260833" \
  VARIANTS="${VARIANT}" \
  PROTECT_RECENT_TOKENS=256 \
  bash "${SCRIPT}" >"${recent256}/launcher.log" 2>&1 &
pid256=$!

env \
  ROOT="${ROOT}" \
  RUN_ROOT="${ordinary}" \
  GPU_IDS="4,5" \
  TOPICS="mixed:20260836" \
  VARIANTS="${VARIANT}" \
  PROTECT_RECENT_TOKENS=0 \
  bash "${SCRIPT}" >"${ordinary}/launcher_logs/mixed_only.log" 2>&1 &
pidmixed=$!

status=0
wait "${pid128}" || status=$?
wait "${pid256}" || status=$?
wait "${pidmixed}" || status=$?
if [[ "${status}" -eq 0 ]]; then
  touch "${ROOT}/results/20260801_qksieve_recent_quota_and_mixed_6gpu_complete"
fi
exit "${status}"
