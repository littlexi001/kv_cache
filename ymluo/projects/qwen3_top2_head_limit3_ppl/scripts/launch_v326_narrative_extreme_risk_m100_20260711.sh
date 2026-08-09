#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-100}"
GPUS="${GPUS:-0,2,3,6}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_v326_narrative_extreme_risk_m100}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

nohup env \
  GPUS="$GPUS" \
  SAMPLES="$SAMPLES" \
  LABEL="v326_narrative_extreme_risk_narrativeqa" \
  STAMP="$STAMP" \
  POLICY="configs/riskkv_task_policy_v326_narrative_extreme_risk_20260711.json" \
  TASKS="narrativeqa" \
  GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-3000}" \
  GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}" \
  bash "$RUNNER" > "$LOG_DIR/launch_v326_narrative_extreme_risk_narrativeqa_${STAMP}_m${SAMPLES}.log" 2>&1 &

echo "Submitted v326 narrative extreme-risk M100 job."
