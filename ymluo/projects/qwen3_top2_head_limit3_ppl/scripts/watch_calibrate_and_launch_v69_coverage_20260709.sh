#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
INTERVAL=${INTERVAL:-120}
LOG=${LOG:-logs/watch_calibrate_and_launch_v69_coverage_20260709.log}
LAUNCH_LOCK=${LAUNCH_LOCK:-logs/riskkv_gpu_launch.lock}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}

BASE_DIR=${BASE_DIR:-outputs/riskkv_v66_task_coverage_mmr_benefit_conformal_qasper_full_m20_20260709}
REFERENCE_DIR=${REFERENCE_DIR:-outputs/riskkv_v64_benefit_conformal_counterfactual_qasper_full_m20_20260709}
BASE_POLICY=${BASE_POLICY:-configs/riskkv_task_policy_v66_task_coverage_mmr_benefit_conformal_qasper_full_20260709.json}
CALIB_CSV=${CALIB_CSV:-outputs/coverage_risk_calibration_v66_vs_v64_m20_20260709.csv}
CALIB_JSON=${CALIB_JSON:-outputs/coverage_risk_calibration_v66_vs_v64_m20_20260709.json}
V69_POLICY=${V69_POLICY:-configs/riskkv_task_policy_v69_calibrated_coverage_mmr_qasper_full_20260709.json}
V69_OUTPUT=${V69_OUTPUT:-outputs/riskkv_v69_calibrated_coverage_mmr_qasper_full_m20_20260709}

cd "$ROOT"
mkdir -p logs outputs

ready() {
  [[ -f "$BASE_DIR/summary.csv" ]] && [[ -f "$REFERENCE_DIR/summary.csv" ]]
}

v69_done_or_running() {
  [[ -f "$V69_OUTPUT/summary.csv" ]] ||
    ps -eo args= | grep -F "run_controlled_public_kv_benchmark_v1.py" | grep -F "$V69_OUTPUT" >/dev/null 2>&1
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
  echo "[$(date)] v69 calibrated coverage watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if v69_done_or_running; then
      echo "[$(date)] v69 already done or running"
      exit 0
    fi
    if ! ready; then
      echo "[$(date)] loop=${i} waiting for base/reference summaries"
      sleep "$INTERVAL"
      continue
    fi
    if [[ ! -f "$V69_POLICY" ]]; then
      "$PY" scripts/calibrate_coverage_risk_gate_20260709.py \
        --base "$BASE_DIR" \
        --reference "$REFERENCE_DIR" \
        --target_recall 0.80 \
        --min_gain 0.01 \
        --min_terms 3 \
        --out_csv "$CALIB_CSV" \
        --out_json "$CALIB_JSON"
      "$PY" scripts/make_coverage_calibrated_policy_20260709.py \
        --base_policy "$BASE_POLICY" \
        --calibration_csv "$CALIB_CSV" \
        --out "$V69_POLICY" \
        --min_beneficial 1 \
        --max_trigger_rate 0.75 \
        --default_budget 2048 \
        --min_terms 3
      echo "[$(date)] generated $V69_POLICY from $CALIB_CSV"
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
      GPU="${gpus[0]}" POLICY="$V69_POLICY" bash scripts/run_riskkv_v69_calibrated_coverage_m20_20260709.sh
      echo "[$(date)] launched v69 on GPU ${gpus[0]}"
      sleep 45
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] watcher timed out without launching v69"
} >> "$LOG" 2>&1
