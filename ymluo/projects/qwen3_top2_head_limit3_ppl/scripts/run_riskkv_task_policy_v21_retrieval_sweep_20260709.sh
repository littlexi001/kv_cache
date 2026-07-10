#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

SAMPLES="${SAMPLES:-10}"
TASKS="${TASKS:-passage_count,passage_retrieval_en,qasper,musique}"
STAMP="${STAMP:-20260709_task_policy_v21_retrieval_sweep}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

run_one() {
  local label="$1"
  local policy="$2"
  LABEL="$label" \
  POLICY="$policy" \
  SAMPLES="$SAMPLES" \
  TASKS="$TASKS" \
  STAMP="$STAMP" \
  bash scripts/run_riskkv_task_policy_v19_one_20260709.sh
}

run_one "retrieval_budget1024" "configs/riskkv_task_policy_v21_retrieval_budget1024_20260709.json"
run_one "retrieval_budget1536" "configs/riskkv_task_policy_v21_retrieval_budget1536_20260709.json"
run_one "retrieval_budget2048" "configs/riskkv_task_policy_v21_retrieval_budget2048_20260709.json"
run_one "retrieval_fullfallback" "configs/riskkv_task_policy_v20_budget_retrieval_safe_20260709.json"
