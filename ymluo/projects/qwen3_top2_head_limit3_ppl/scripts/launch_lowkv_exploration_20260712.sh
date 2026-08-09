#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260712_lowkv_exploration}"
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
  echo "LAUNCH label=${label} samples=${samples} tasks=${tasks} policy=${policy} log=${log}"
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

launch_one "v360_lowkv_certificate_all" "configs/riskkv_task_policy_v360_lowkv_certificate_20260712.json" "$ALL_TASKS" 20
launch_one "v361_lowkv_graph_bridge_hard" "configs/riskkv_task_policy_v361_lowkv_graph_bridge_20260712.json" "$HARD_TASKS" 40
launch_one "v362_lowkv_bounded_retry_hard" "configs/riskkv_task_policy_v362_lowkv_bounded_retry_20260712.json" "$HARD_TASKS" 40
launch_one "v363_taskwise_lowkv_mix_all" "configs/riskkv_task_policy_v363_taskwise_lowkv_mix_20260712.json" "$ALL_TASKS" 20
launch_one "v364_extreme_hardtask_probe_hard" "configs/riskkv_task_policy_v364_extreme_hardtask_probe_20260712.json" "$HARD_TASKS" 40
launch_one "v365_ultra_skeleton_all" "configs/riskkv_task_policy_v365_ultra_skeleton_all_20260712.json" "$ALL_TASKS" 20
launch_one "v366_skeleton_support_retry_hard" "configs/riskkv_task_policy_v366_skeleton_support_retry_20260712.json" "$HARD_TASKS" 40
launch_one "v367_query_only_then_verify_hard" "configs/riskkv_task_policy_v367_query_only_then_verify_20260712.json" "$HARD_TASKS" 40
launch_one "v368_direct_operator_extreme_mix_all" "configs/riskkv_task_policy_v368_direct_operator_extreme_mix_20260712.json" "$ALL_TASKS" 20
launch_one "v369_hardtask_minimal_ablation_hard" "configs/riskkv_task_policy_v369_hardtask_minimal_ablation_20260712.json" "$HARD_TASKS" 40

echo "Submitted low-KV exploration jobs at $(date -Is)."
