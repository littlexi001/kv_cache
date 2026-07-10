#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_v67_v68_after_v66_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

v66_done() {
  [[ -f outputs/riskkv_v66_task_coverage_mmr_benefit_conformal_qasper_full_m20_20260709/summary.csv ]]
}

done_or_running() {
  local name=$1
  [[ -f "outputs/${name}/summary.csv" ]] || pgrep -af "$name" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

{
  echo "[$(date)] v67/v68 watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if done_or_running "riskkv_v67_coverage_risk_gate_qasper_full_m20_20260709" &&
      done_or_running "riskkv_v68_coverage_mmr_risk_gate_qasper_full_m20_20260709"; then
      echo "[$(date)] v67/v68 already done or running"
      exit 0
    fi
    if ! v66_done; then
      echo "[$(date)] loop=${i} waiting for v66 summary"
      sleep "$INTERVAL"
      continue
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=${i} free_gpus=${gpus[*]:-none}"
    gpu_index=0
    if ! done_or_running "riskkv_v67_coverage_risk_gate_qasper_full_m20_20260709" &&
      (( ${#gpus[@]} > gpu_index )); then
      GPU="${gpus[$gpu_index]}" \
        NAME="riskkv_v67_coverage_risk_gate_qasper_full_m20_20260709" \
        POLICY="configs/riskkv_task_policy_v67_coverage_risk_gate_qasper_full_20260709.json" \
        bash scripts/run_riskkv_single_policy_m20_20260709.sh
      echo "[$(date)] launched v67 on GPU ${gpus[$gpu_index]}"
      gpu_index=$((gpu_index + 1))
    fi
    if ! done_or_running "riskkv_v68_coverage_mmr_risk_gate_qasper_full_m20_20260709" &&
      (( ${#gpus[@]} > gpu_index )); then
      GPU="${gpus[$gpu_index]}" \
        NAME="riskkv_v68_coverage_mmr_risk_gate_qasper_full_m20_20260709" \
        POLICY="configs/riskkv_task_policy_v68_coverage_mmr_risk_gate_qasper_full_20260709.json" \
        bash scripts/run_riskkv_single_policy_m20_20260709.sh
      echo "[$(date)] launched v68 on GPU ${gpus[$gpu_index]}"
      gpu_index=$((gpu_index + 1))
    fi
    if (( gpu_index > 0 )); then
      echo "[$(date)] launched missing v67/v68 jobs"
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching v67/v68"
} >> "$LOG" 2>&1
