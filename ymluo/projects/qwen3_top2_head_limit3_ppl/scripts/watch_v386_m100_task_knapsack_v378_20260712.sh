#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-0,1}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260712_m100_task_knapsack_v378_v386}"
LABEL="${LABEL:-v386_m100_task_knapsack_v378}"
POLICY="${POLICY:-configs/riskkv_task_policy_v386_m100_task_knapsack_v378_20260712.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
ALL_TASKS="${ALL_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
mkdir -p "$LOG_DIR"

launch_sync() {
  local samples="$1"
  local out="outputs/riskkv_v19_${LABEL}_${STAMP}_m${samples}_bDyn_pDyn"
  local log="$LOG_DIR/launch_${LABEL}_${STAMP}_m${samples}.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing ${out}/task_results.csv $(date -Is)"
    return 0
  fi
  echo "RUN label=${LABEL} samples=${samples} policy=${POLICY} $(date -Is)"
  env \
    GPUS="$GPUS" \
    LOCK_ROOT="$LOCK_ROOT" \
    SAMPLES="$samples" \
    LABEL="$LABEL" \
    STAMP="$STAMP" \
    POLICY="$POLICY" \
    TASKS="$ALL_TASKS" \
    bash "$RUNNER" > "$log" 2>&1
}

launch_sync 20
launch_sync 100

"$PY" scripts/summarize_lowkv_exploration_20260712.py || true
"$PY" scripts/parse_lowkv_running_progress_20260712.py || true
