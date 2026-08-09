#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

RUNNER="scripts/run_riskkv_task_policy_v19_one_20260709.sh"
TASKS="gov_report,qmsum,multi_news,trec,samsum,passage_count,passage_retrieval_en"
SAMPLES=10
STAMP="20260712_purefront"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

# label|budget|page|sink|recent
ACTIONS=(
  "b128_p16|128|16|32|32"
  "b256_p16|256|16|32|32"
  "b256_p64|256|64|32|32"
  "b512_p64|512|64|64|64"
  "b512_p128|512|128|64|64"
  "b1024_p128|1024|128|64|128"
  "b2048_p256|2048|256|64|128"
)

run_action() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r label budget page sink recent <<< "$spec"
  local policy
  policy="{\"__runtime_constraints\":{\"direct_structured_answer\":false},\"*\":{\"budget_tokens\":${budget},\"page_tokens\":${page},\"sink_tokens\":${sink},\"recent_tokens\":${recent},\"scorer\":\"hybrid_late_mmr_multiscale_flow\",\"short_decode\":false,\"direct_structured_answer\":false}}"
  local log="$LOG_ROOT/nohup_v441_purefront_${label}_m10_20260712.log"
  echo "START v441 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
  GPUS="$gpu" SAMPLES="$SAMPLES" TASKS="$TASKS" LABEL="v441_purefront_${label}" \
    STAMP="$STAMP" POLICY="$policy" bash "$RUNNER" >> "$log" 2>&1
  echo "DONE v441 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
}

worker() {
  local worker_id="$1"
  local gpu="$2"
  local index
  for index in "${!ACTIONS[@]}"; do
    if (( index % 2 == worker_id )); then
      run_action "$gpu" "${ACTIONS[$index]}"
    fi
  done
}

worker 0 6 &
pid0=$!
worker 1 7 &
pid1=$!
wait "$pid0"
wait "$pid1"
echo "V441 pure action frontier complete $(date -Is)"
