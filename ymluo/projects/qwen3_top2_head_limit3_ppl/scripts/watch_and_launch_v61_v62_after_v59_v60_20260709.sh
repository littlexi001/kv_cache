#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v61_v62_after_v59_v60_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

prereq_done() {
  [[ -f outputs/riskkv_v59_selective_counterfactual_m20_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v60_selective_counterfactual_qasper_full_m20_20260709/summary.csv ]]
}

already_done() {
  [[ -f outputs/riskkv_v61_conformal_counterfactual_m20_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v62_conformal_counterfactual_qasper_full_m20_20260709/summary.csv ]]
}

already_running() {
  pgrep -af "riskkv_v6[12]_conformal_counterfactual.*m20_20260709" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

{
  echo "[$(date)] watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if already_done; then
      echo "[$(date)] v61/v62 conformal-counterfactual outputs already complete"
      exit 0
    fi
    if already_running; then
      echo "[$(date)] v61/v62 conformal-counterfactual already running"
      exit 0
    fi
    if ! prereq_done; then
      echo "[$(date)] loop=$i waiting for v59/v60 selective-counterfactual summaries"
      sleep "$INTERVAL"
      continue
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=$i free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 2 )); then
      GPU_V61=${gpus[0]} GPU_V62=${gpus[1]} bash scripts/run_riskkv_v61_v62_conformal_counterfactual_m20_20260709.sh
      echo "[$(date)] launched v61/v62 on GPUs ${gpus[0]} ${gpus[1]}"
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching"
} >> "$LOG" 2>&1
