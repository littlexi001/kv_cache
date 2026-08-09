#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-100}"
GPUS="${GPUS:-0,2,3,6}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_v324_qasper_narrative_nofull_m100}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label="$1"
  local task="$2"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${SAMPLES}.log"
  echo "LAUNCH label=${label} task=${task} samples=${SAMPLES} gpus=${GPUS} log=${log}"
  nohup env \
    GPUS="$GPUS" \
    SAMPLES="$SAMPLES" \
    LABEL="$label" \
    STAMP="$STAMP" \
    POLICY="configs/riskkv_task_policy_v324_qasper_narrative_nofull_20260711.json" \
    TASKS="$task" \
    GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-3000}" \
    GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}" \
    bash "$RUNNER" > "$log" 2>&1 &
}

for task in narrativeqa qasper; do
  launch_one "v324_qasper_narrative_nofull_${task}" "$task"
done

echo "Submitted v324 qasper+narrative no-full M100 jobs."
