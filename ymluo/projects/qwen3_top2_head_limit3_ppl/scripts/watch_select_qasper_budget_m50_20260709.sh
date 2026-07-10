#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_select_qasper_budget_m50_20260709.log}
LAUNCH_LOCK=${LAUNCH_LOCK:-logs/riskkv_gpu_launch.lock}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

required_summaries() {
  for dir in \
    outputs/riskkv_v81_v72_qasper_budgeted_m20_20260709 \
    outputs/riskkv_v82_v72_qasper1024_m20_retry_20260709 \
    outputs/riskkv_v83_v72_qasper1536_m20_retry_20260709 \
    outputs/riskkv_v84_v72_qasper3072_m20_retry_20260709 \
    outputs/riskkv_v85_v72_qasper_adaptive1024_2048_m20_20260709; do
    [[ -f "${dir}/summary.csv" ]] || return 1
  done
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

done_or_running() {
  local name="$1"
  [[ -f "outputs/${name}/summary.csv" ]] ||
    ps -eo args= | grep -F "run_controlled_public_kv_benchmark_v1.py" | grep -F "outputs/${name}" >/dev/null 2>&1
}

{
  echo "[$(date)] qasper budget m50 selector watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if ! required_summaries; then
      echo "[$(date)] loop=${i} waiting for qasper m20 summaries"
      sleep "$INTERVAL"
      continue
    fi

    eval "$("$PY" scripts/select_qasper_budget_policy_20260709.py --shell)"
    echo "[$(date)] selected label=${SELECTED_LABEL} policy=${SELECTED_POLICY} m50=${SELECTED_M50_NAME} score=${SELECTED_SCORE} keep=${SELECTED_KEEP} reason=${SELECTION_REASON}"

    if done_or_running "$SELECTED_M50_NAME"; then
      echo "[$(date)] selected m50 already done or running: ${SELECTED_M50_NAME}"
      exit 0
    fi

    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} < 1 )); then
      sleep "$INTERVAL"
      continue
    fi

    acquire_launch_lock
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] locked free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} < 1 )); then
      rmdir "$LAUNCH_LOCK" 2>/dev/null || true
      trap - EXIT
      sleep "$INTERVAL"
      continue
    fi
    GPU="${gpus[0]}" MAX_SAMPLES=50 NAME="$SELECTED_M50_NAME" POLICY="$SELECTED_POLICY" \
      bash scripts/run_riskkv_single_policy_m20_20260709.sh
    echo "[$(date)] launched selected qasper m50 ${SELECTED_M50_NAME} on GPU ${gpus[0]}"
    sleep 45
    exit 0
  done
  echo "[$(date)] watcher timed out before selecting qasper m50"
} >> "$LOG" 2>&1
