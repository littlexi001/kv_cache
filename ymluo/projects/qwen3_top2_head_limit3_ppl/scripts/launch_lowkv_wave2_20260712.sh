#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260712_lowkv_extreme_1to10}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
mkdir -p "$LOG_DIR"

ALL_TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
HARD_TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,repobench-p"

launch_one() {
  local label="$1"
  local policy="$2"
  local tasks="$3"
  local samples="$4"
  local log="$LOG_DIR/launch_${label}_${STAMP}_m${samples}.log"
  echo "LAUNCH wave2 label=${label} samples=${samples} tasks=${tasks} policy=${policy} log=${log}"
  nohup env \
    GPUS="$GPUS" \
    LOCK_ROOT="$LOCK_ROOT" \
    SAMPLES="$samples" \
    LABEL="$label" \
    STAMP="$STAMP" \
    POLICY="$policy" \
    TASKS="$tasks" \
    bash "$RUNNER" > "$log" 2>&1 &
}

launch_one "v370_lowkv_uncertainty_ladder_all" "configs/riskkv_task_policy_v370_lowkv_uncertainty_ladder_20260712.json" "$ALL_TASKS" 20
launch_one "v371_lowkv_span_repack_ladder_hard" "configs/riskkv_task_policy_v371_lowkv_span_repack_ladder_20260712.json" "$HARD_TASKS" 40
launch_one "v372_extractive_qa_direct_all" "configs/riskkv_task_policy_v372_extractive_qa_direct_20260712.json" "$ALL_TASKS" 20
launch_one "v373_selective_direct_ladder_all" "configs/riskkv_task_policy_v373_selective_direct_ladder_20260712.json" "$ALL_TASKS" 20
launch_one "v374_lowkv_verifier_retry_hard" "configs/riskkv_task_policy_v374_lowkv_verifier_retry_20260712.json" "$HARD_TASKS" 40

echo "Submitted low-KV wave2 jobs at $(date -Is)."
