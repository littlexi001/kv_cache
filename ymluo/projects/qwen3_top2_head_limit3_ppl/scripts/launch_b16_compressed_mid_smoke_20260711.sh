#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-20}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_b16_compressed_mid_smoke}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

launch_one() {
  local label="$1"
  local policy="$2"
  local task="$3"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${SAMPLES}.log"
  local out="outputs/riskkv_v19_${label}_${STAMP}_m${SAMPLES}_bDyn_pDyn"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP_DONE label=${label} task=${task} out=${out}"
    return
  fi
  echo "LAUNCH label=${label} task=${task} policy=${policy} samples=${SAMPLES} log=${log}"
  nohup env \
    GPUS="$GPUS" \
    GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-3000}" \
    GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}" \
    SAMPLES="$SAMPLES" \
    LABEL="$label" \
    STAMP="$STAMP" \
    POLICY="$policy" \
    TASKS="$task" \
    bash "$RUNNER" > "$log" 2>&1 &
}

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v327_b16_compressed_mid_${task}" \
    "configs/riskkv_task_policy_v327_b16_compressed_mid_smoke_20260711.json" \
    "$task"
done

for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
  launch_one \
    "v328_b16_window64_compressed_mid_${task}" \
    "configs/riskkv_task_policy_v328_b16_window64_compressed_mid_smoke_20260711.json" \
    "$task"
done

echo "Submitted B16 compressed-mid smoke jobs."
