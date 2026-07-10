#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

M6_STAMP="${M6_STAMP:-20260709_task_policy_v19_fixfull}"
M20_STAMP="${M20_STAMP:-20260709_task_policy_v19_fixfull_m20}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

safe_m6="outputs/riskkv_v19_safe_${M6_STAMP}_m6_bDyn_pDyn/summary.csv"
budget_m6="outputs/riskkv_v19_budget_${M6_STAMP}_m6_bDyn_pDyn/summary.csv"
safe_m20="outputs/riskkv_v19_safe_${M20_STAMP}_m20_bDyn_pDyn/summary.csv"
budget_m20="outputs/riskkv_v19_budget_${M20_STAMP}_m20_bDyn_pDyn/summary.csv"

while [[ ! -f "$safe_m6" || ! -f "$budget_m6" ]]; do
  echo "WAIT_V19_M6 safe=$([[ -f "$safe_m6" ]] && echo yes || echo no) budget=$([[ -f "$budget_m6" ]] && echo yes || echo no) $(date -Is)"
  sleep 300
done

echo "FOUND_V19_M6 $(date -Is)"
grep '^longbench,ALL' "$safe_m6" || true
grep '^longbench,ALL' "$budget_m6" || true

if [[ -f "$safe_m20" && -f "$budget_m20" ]]; then
  echo "V19_M20_ALREADY_DONE $(date -Is)"
  exit 0
fi

echo "START_V19_M20 stamp=$M20_STAMP $(date -Is)"
SAMPLES=20 STAMP="$M20_STAMP" bash scripts/run_riskkv_task_policy_v19_20260709.sh
echo "DONE_V19_M20 $(date -Is)"
