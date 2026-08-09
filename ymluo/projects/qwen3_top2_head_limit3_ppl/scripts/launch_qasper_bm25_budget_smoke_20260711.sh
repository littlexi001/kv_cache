#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-20}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_qasper_bm25_budget_smoke}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label="$1"
  local policy="$2"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${SAMPLES}.log"
  echo "LAUNCH label=${label} policy=${policy} samples=${SAMPLES} log=${log}"
  nohup env \
    GPUS="$GPUS" \
    SAMPLES="$SAMPLES" \
    LABEL="$label" \
    STAMP="$STAMP" \
    POLICY="$policy" \
    TASKS="qasper" \
    bash "$RUNNER" > "$log" 2>&1 &
}

launch_one \
  "v318_qasper_b128_bm25bridge_1280" \
  "configs/riskkv_task_policy_v318_qasper_b128_bm25bridge_1280_smoke_20260711.json"

launch_one \
  "v319_qasper_b128_bm25bridge_1024" \
  "configs/riskkv_task_policy_v319_qasper_b128_bm25bridge_1024_smoke_20260711.json"

echo "Submitted qasper BM25 budget smoke jobs."
