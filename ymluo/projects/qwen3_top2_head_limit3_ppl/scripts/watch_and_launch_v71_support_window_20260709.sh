#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v71_support_window_20260709.log}
LAUNCH_LOCK=${LAUNCH_LOCK:-logs/riskkv_gpu_launch.lock}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}
NAME=${NAME:-riskkv_v71_support_window_qa_verifier_qasper_full_m20_20260709}
POLICY=${POLICY:-configs/riskkv_task_policy_v71_support_window_qa_verifier_qasper_full_20260709.json}

cd "$ROOT"
mkdir -p logs

done_or_running() {
  [[ -f "outputs/${NAME}/summary.csv" ]] ||
    ps -eo args= | grep -F "run_controlled_public_kv_benchmark_v1.py" | grep -F "outputs/${NAME}" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

acquire_launch_lock() {
  while ! mkdir "$LAUNCH_LOCK" 2>/dev/null; do
    echo "[$(date)] waiting for launch lock $LAUNCH_LOCK"
    sleep 15
  done
  trap 'rmdir "$LAUNCH_LOCK" 2>/dev/null || true' EXIT
}

{
  echo "[$(date)] v71 support-window watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if done_or_running; then
      echo "[$(date)] ${NAME} already done or running"
      exit 0
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=${i} free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 1 )); then
      acquire_launch_lock
      mapfile -t gpus < <(free_gpus)
      echo "[$(date)] locked free_gpus=${gpus[*]:-none}"
      if (( ${#gpus[@]} < 1 )); then
        rmdir "$LAUNCH_LOCK" 2>/dev/null || true
        trap - EXIT
        sleep "$INTERVAL"
        continue
      fi
      GPU="${gpus[0]}" MAX_SAMPLES=20 NAME="$NAME" POLICY="$POLICY" \
        bash scripts/run_riskkv_single_policy_m20_20260709.sh
      echo "[$(date)] launched ${NAME} on GPU ${gpus[0]}"
      sleep 45
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching ${NAME}"
} >> "$LOG" 2>&1
