#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-100}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_b16_group_sweep}"
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

# Complete the missing no-window high-recall hotpot point from v289.
launch_one \
  "v289_b16_highrecall_hotpotqa" \
  "configs/riskkv_task_policy_v289_v286_b16_highrecall_20260711.json" \
  "hotpotqa"

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v301_b16_group4_highrecall_${task}" \
    "configs/riskkv_task_policy_v301_b16_group4_highrecall_20260711.json" \
    "$task"
done

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v302_b16_group2_highrecall_${task}" \
    "configs/riskkv_task_policy_v302_b16_group2_highrecall_20260711.json" \
    "$task"
done

echo "Submitted b16 group-size sweep jobs."
