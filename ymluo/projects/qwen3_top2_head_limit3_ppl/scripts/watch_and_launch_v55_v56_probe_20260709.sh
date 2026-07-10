#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v55_v56_probe_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

already_done() {
  [[ -f outputs/riskkv_v55_consistency_quality_probe16_m20_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v56_consistency_quality_qasper_full_probe16_m20_20260709/summary.csv ]]
}

already_running() {
  pgrep -af "riskkv_v5[56]_consistency.*probe16_m20_20260709" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

{
  echo "[$(date)] watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if already_done; then
      echo "[$(date)] v55/v56 probe outputs already complete"
      exit 0
    fi
    if already_running; then
      echo "[$(date)] v55/v56 probe already running"
      exit 0
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=$i free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 2 )); then
      GPU_V55=${gpus[0]} GPU_V56=${gpus[1]} bash scripts/run_riskkv_v55_v56_probe_m20_20260709.sh
      echo "[$(date)] launched v55/v56 on GPUs ${gpus[0]} ${gpus[1]}"
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching"
} >> "$LOG" 2>&1
