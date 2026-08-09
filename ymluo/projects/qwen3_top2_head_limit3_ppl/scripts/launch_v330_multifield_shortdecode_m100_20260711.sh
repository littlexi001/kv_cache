#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

SAMPLES="${SAMPLES:-100}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260711_v330_multifield_shortdecode_m100}"
LABEL="${LABEL:-v330_multifield_shortdecode}"
POLICY="${POLICY:-configs/riskkv_task_policy_v317_v300_qa_shortdecode_aggressive_smoke_20260711.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

log="$LOG_DIR/launch_${LABEL}_${STAMP}_m${SAMPLES}.log"
out="outputs/riskkv_v19_${LABEL}_${STAMP}_m${SAMPLES}_bDyn_pDyn"
if [[ -f "$out/task_results.csv" ]]; then
  echo "SKIP_DONE label=${LABEL} out=${out}"
  exit 0
fi

nohup env \
  GPUS="$GPUS" \
  GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-3000}" \
  GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}" \
  SAMPLES="$SAMPLES" \
  LABEL="$LABEL" \
  STAMP="$STAMP" \
  POLICY="$POLICY" \
  TASKS="multifieldqa_en" \
  bash "$RUNNER" > "$log" 2>&1 &

echo "LAUNCHED pid=$! label=${LABEL} samples=${SAMPLES} log=${log}"
