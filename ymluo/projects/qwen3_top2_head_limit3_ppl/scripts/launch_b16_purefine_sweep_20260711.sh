#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-100}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_b16_purefine_sweep}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label="$1"
  local policy="$2"
  local task="$3"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${SAMPLES}.log"
  echo "LAUNCH label=${label} task=${task} policy=${policy} samples=${SAMPLES} log=${log}"
  nohup env \
    GPUS="$GPUS" \
    SAMPLES="$SAMPLES" \
    LABEL="$label" \
    STAMP="$STAMP" \
    POLICY="$policy" \
    TASKS="$task" \
    bash "$RUNNER" > "$log" 2>&1 &
}

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v307_b16_purefine_highrecall_${task}" \
    "configs/riskkv_task_policy_v307_b16_purefine_highrecall_20260711.json" \
    "$task"
done

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v308_b16_purefine_window_highrecall_${task}" \
    "configs/riskkv_task_policy_v308_b16_purefine_window_highrecall_20260711.json" \
    "$task"
done

echo "Submitted b16 pure-fine sweep jobs."
