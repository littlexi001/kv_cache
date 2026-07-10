#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-180}
LOG=${LOG:-logs/watch_and_launch_m100_after_m50_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

m50_ready() {
  [[ -f outputs/riskkv_fullkv_m50_same_samples_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v37_high_quality_m50_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v52_consistency_quality_m50_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v53_consistency_quality_qasper_full_m50_20260709/summary.csv ]]
}

m100_done_or_running() {
  [[ -f outputs/riskkv_v53_consistency_quality_qasper_full_m100_20260709/summary.csv ]] ||
    pgrep -af "riskkv_.*m100_20260709" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

{
  echo "[$(date)] m100 watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if m100_done_or_running; then
      echo "[$(date)] m100 already done or running"
      exit 0
    fi
    if ! m50_ready; then
      echo "[$(date)] loop=$i m50 not ready"
      sleep "$INTERVAL"
      continue
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=$i m50 ready free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 4 )); then
      GPU_FULL=${gpus[0]} GPU_V37=${gpus[1]} GPU_V52=${gpus[2]} GPU_V53=${gpus[3]} \
        bash scripts/run_riskkv_v37_v52_v53_m100_20260709.sh
      echo "[$(date)] launched m100 on GPUs ${gpus[0]} ${gpus[1]} ${gpus[2]} ${gpus[3]}"
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] m100 watcher timed out without launching"
} >> "$LOG" 2>&1
