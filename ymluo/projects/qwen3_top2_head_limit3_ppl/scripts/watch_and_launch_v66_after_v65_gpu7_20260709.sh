#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v66_after_v65_gpu7_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}
GPU=${GPU:-7}

cd "$ROOT"
mkdir -p logs

v65_done() {
  [[ -f outputs/riskkv_v65_coverage_mmr_benefit_conformal_qasper_full_m20_20260709/summary.csv ]]
}

v66_done_or_running() {
  [[ -f outputs/riskkv_v66_task_coverage_mmr_benefit_conformal_qasper_full_m20_20260709/summary.csv ]] ||
    pgrep -af "riskkv_v66_task_coverage_mmr.*m20_20260709" >/dev/null 2>&1
}

gpu_free() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($1 + 0) < 1000 && ($2 + 0) < 20) ok=1} END {exit ok ? 0 : 1}'
}

{
  echo "[$(date)] v66-after-v65 watcher started on GPU ${GPU}"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if v66_done_or_running; then
      echo "[$(date)] v66 already done or running"
      exit 0
    fi
    if ! v65_done; then
      echo "[$(date)] loop=${i} waiting for v65 summary"
      sleep "$INTERVAL"
      continue
    fi
    if gpu_free; then
      GPU="$GPU" bash scripts/run_riskkv_v66_task_coverage_mmr_m20_20260709.sh
      echo "[$(date)] launched v66 on GPU ${GPU}"
      exit 0
    fi
    echo "[$(date)] loop=${i} v65 done but GPU ${GPU} not free"
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching v66"
} >> "$LOG" 2>&1
