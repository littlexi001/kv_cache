#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_and_launch_riskkv_priority_queue_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

cd "$ROOT"
mkdir -p logs

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

pair_done() {
  local first=$1
  local second=$2
  [[ -f "outputs/${first}/summary.csv" ]] && [[ -f "outputs/${second}/summary.csv" ]]
}

pair_running() {
  local pattern=$1
  pgrep -af "$pattern" >/dev/null 2>&1
}

wait_for_pair_done() {
  local label=$1
  local first=$2
  local second=$3
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if pair_done "$first" "$second"; then
      echo "[$(date)] prerequisite complete: ${label}"
      return 0
    fi
    echo "[$(date)] waiting for ${label}: loop=${i}"
    sleep "$INTERVAL"
  done
  echo "[$(date)] timed out waiting for ${label}" >&2
  return 1
}

launch_pair_when_free() {
  local label=$1
  local first=$2
  local second=$3
  local running_pattern=$4
  local run_script=$5
  local env_first=$6
  local env_second=$7

  if pair_done "$first" "$second"; then
    echo "[$(date)] ${label} already complete"
    return 0
  fi
  if pair_running "$running_pattern"; then
    echo "[$(date)] ${label} already running"
    wait_for_pair_done "$label" "$first" "$second"
    return 0
  fi

  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] ${label} launch loop=${i} free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 2 )); then
      env "$env_first=${gpus[0]}" "$env_second=${gpus[1]}" bash "$run_script"
      echo "[$(date)] launched ${label} on GPUs ${gpus[0]} ${gpus[1]}"
      sleep 60
      wait_for_pair_done "$label" "$first" "$second"
      return 0
    fi
    sleep "$INTERVAL"
  done

  echo "[$(date)] timed out before launching ${label}" >&2
  return 1
}

{
  echo "[$(date)] RiskKV priority queue watcher started"

  wait_for_pair_done \
    "v55/v56 probe" \
    "riskkv_v55_consistency_quality_probe16_m20_20260709" \
    "riskkv_v56_consistency_quality_qasper_full_probe16_m20_20260709"

  launch_pair_when_free \
    "v63/v64 benefit-conformal" \
    "riskkv_v63_benefit_conformal_counterfactual_m20_20260709" \
    "riskkv_v64_benefit_conformal_counterfactual_qasper_full_m20_20260709" \
    "riskkv_v6[34]_benefit_conformal.*m20_20260709" \
    "scripts/run_riskkv_v63_v64_benefit_conformal_m20_20260709.sh" \
    "GPU_V63" \
    "GPU_V64"

  launch_pair_when_free \
    "v57/v58 predecode score-risk" \
    "riskkv_v57_predecode_score_risk_full_m20_20260709" \
    "riskkv_v58_predecode_score_risk_2048_m20_20260709" \
    "riskkv_v5[78]_predecode_score_risk.*m20_20260709" \
    "scripts/run_riskkv_v57_v58_score_risk_m20_20260709.sh" \
    "GPU_V57" \
    "GPU_V58"

  launch_pair_when_free \
    "v59/v60 selective counterfactual" \
    "riskkv_v59_selective_counterfactual_m20_20260709" \
    "riskkv_v60_selective_counterfactual_qasper_full_m20_20260709" \
    "riskkv_v[56][90]_selective_counterfactual.*m20_20260709" \
    "scripts/run_riskkv_v59_v60_selective_counterfactual_m20_20260709.sh" \
    "GPU_V59" \
    "GPU_V60"

  launch_pair_when_free \
    "v61/v62 conformal counterfactual" \
    "riskkv_v61_conformal_counterfactual_m20_20260709" \
    "riskkv_v62_conformal_counterfactual_qasper_full_m20_20260709" \
    "riskkv_v6[12]_conformal_counterfactual.*m20_20260709" \
    "scripts/run_riskkv_v61_v62_conformal_counterfactual_m20_20260709.sh" \
    "GPU_V61" \
    "GPU_V62"

  echo "[$(date)] RiskKV priority queue watcher complete"
} >> "$LOG" 2>&1
