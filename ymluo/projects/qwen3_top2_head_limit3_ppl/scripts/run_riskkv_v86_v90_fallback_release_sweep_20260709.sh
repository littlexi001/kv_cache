#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

SAMPLES="${SAMPLES:-20}"
TASKS="${TASKS:-hotpotqa,musique,trec,passage_count,repobench-p,qasper}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAMP="${STAMP:-20260709_fallback_release_sweep}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

launch_one() {
  local label="$1"
  local policy="$2"
  local launch_log="$LOG_ROOT/launch_${label}_${STAMP}_m${SAMPLES}.log"
  echo "LAUNCH label=$label policy=$policy samples=$SAMPLES tasks=$TASKS $(date -Is)"
  LABEL="$label" \
  POLICY="$policy" \
  SAMPLES="$SAMPLES" \
  TASKS="$TASKS" \
  GPUS="$GPUS" \
  STAMP="$STAMP" \
  bash scripts/run_riskkv_task_policy_v19_one_20260709.sh > "$launch_log" 2>&1 &
  sleep 8
}

launch_one "v86_hotpot2048" "configs/riskkv_task_policy_v86_v81_hotpot_budget2048_20260709.json"
launch_one "v87_musique2048" "configs/riskkv_task_policy_v87_v81_musique_budget2048_20260709.json"
launch_one "v88_hotpot_musique2048" "configs/riskkv_task_policy_v88_v81_hotpot_musique_budget2048_20260709.json"
launch_one "v89_static_release" "configs/riskkv_task_policy_v89_v81_static_tasks_release_20260709.json"
launch_one "v90_full_release_adaptive" "configs/riskkv_task_policy_v90_v81_full_release_adaptive_20260709.json"

wait

echo "DONE fallback release sweep $(date -Is)"
for f in outputs/riskkv_v19_*_${STAMP}_m${SAMPLES}_bDyn_pDyn/summary.csv; do
  echo "SUMMARY $f"
  cat "$f"
done
