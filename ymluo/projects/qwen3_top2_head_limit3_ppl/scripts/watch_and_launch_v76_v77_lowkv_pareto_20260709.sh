#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v76_v77_lowkv_pareto_20260709.log}
LAUNCH_LOCK=${LAUNCH_LOCK:-logs/riskkv_gpu_launch.lock}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

JOBS=(
  "riskkv_v76_lowkv_no_static_full_m20_20260709|configs/riskkv_task_policy_v76_lowkv_no_static_full_20260709.json|20"
  "riskkv_v77_midkv_budgeted_safety_m20_20260709|configs/riskkv_task_policy_v77_midkv_budgeted_safety_20260709.json|20"
)

done_or_running() {
  local name="$1"
  [[ -f "outputs/${name}/summary.csv" ]] ||
    ps -eo args= | grep -F "run_controlled_public_kv_benchmark_v1.py" | grep -F "outputs/${name}" >/dev/null 2>&1
}

next_job() {
  local spec name
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ <<< "$spec"
    if ! done_or_running "$name"; then
      echo "$spec"
      return 0
    fi
  done
  return 1
}

all_done_or_running() {
  local spec name
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ <<< "$spec"
    if ! done_or_running "$name"; then
      return 1
    fi
  done
  return 0
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
}

release_launch_lock() {
  rmdir "$LAUNCH_LOCK" 2>/dev/null || true
}

{
  echo "[$(date)] v76/v77 low-kv Pareto watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if all_done_or_running; then
      echo "[$(date)] all v76/v77 jobs are already done or running"
      exit 0
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=${i} free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 1 )); then
      acquire_launch_lock
      mapfile -t gpus < <(free_gpus)
      echo "[$(date)] locked free_gpus=${gpus[*]:-none}"
      if (( ${#gpus[@]} < 1 )); then
        release_launch_lock
        sleep "$INTERVAL"
        continue
      fi
      spec="$(next_job || true)"
      if [[ -z "${spec:-}" ]]; then
        release_launch_lock
        echo "[$(date)] no pending v76/v77 job"
        exit 0
      fi
      IFS='|' read -r name policy samples <<< "$spec"
      GPU="${gpus[0]}" MAX_SAMPLES="$samples" NAME="$name" POLICY="$policy" \
        bash scripts/run_riskkv_single_policy_m20_20260709.sh
      release_launch_lock
      echo "[$(date)] launched ${name} on GPU ${gpus[0]}"
      sleep 45
      continue
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out before all v76/v77 jobs launched"
} >> "$LOG" 2>&1
