#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
ACTIVE_PID="${ACTIVE_PID:?set ACTIVE_PID to the active 256K Python PID}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_keymse_512k_extrap_budget_frontier_7gpu}"

while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  sleep 60
done

cd "${PROJECT_ROOT}"
RUN_ROOT="${RUN_ROOT}" \
  bash scripts/launch_qksieve_keymse_512k_budget_frontier_7gpu_20260730.sh
