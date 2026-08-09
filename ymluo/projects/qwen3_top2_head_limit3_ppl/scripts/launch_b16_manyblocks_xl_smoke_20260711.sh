#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-20}"
GPUS="${GPUS:-2}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_b16_manyblocks_xl_smoke}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label="$1"
  local policy="$2"
  local task="$3"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${SAMPLES}.log"
  echo "LAUNCH label=${label} task=${task} policy=${policy} samples=${SAMPLES} gpus=${GPUS} log=${log}"
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
    "v320_b16_manyblocks_xl_${task}" \
    "configs/riskkv_task_policy_v320_b16_manyblocks_xl_smoke_20260711.json" \
    "$task"
done

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v321_b16_manyblocks_window64_${task}" \
    "configs/riskkv_task_policy_v321_b16_manyblocks_window64_smoke_20260711.json" \
    "$task"
done

echo "Submitted B16 many-block XL smoke jobs."
