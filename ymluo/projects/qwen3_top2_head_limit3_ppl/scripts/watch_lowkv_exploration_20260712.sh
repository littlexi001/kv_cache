#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

STAMP="${STAMP:-20260712_lowkv_exploration}"
LOG="outputs/logs/watch_lowkv_exploration_${STAMP}.log"
mkdir -p outputs/logs

echo "WATCH lowkv exploration stamp=${STAMP} $(date -Is)" | tee -a "$LOG"

expected=(
  "riskkv_v19_v360_lowkv_certificate_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v361_lowkv_graph_bridge_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v362_lowkv_bounded_retry_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v363_taskwise_lowkv_mix_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v364_extreme_hardtask_probe_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v365_ultra_skeleton_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v366_skeleton_support_retry_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v367_query_only_then_verify_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v368_direct_operator_extreme_mix_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v369_hardtask_minimal_ablation_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v370_lowkv_uncertainty_ladder_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v371_lowkv_span_repack_ladder_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v372_extractive_qa_direct_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v373_selective_direct_ladder_all_${STAMP}_m20_bDyn_pDyn/task_results.csv"
  "riskkv_v19_v374_lowkv_verifier_retry_hard_${STAMP}_m40_bDyn_pDyn/task_results.csv"
)

while true; do
  complete=0
  for path in "${expected[@]}"; do
    if [[ -s "outputs/${path}" ]]; then
      complete=$((complete + 1))
    fi
  done
  echo "COMPLETE ${complete}/${#expected[@]} $(date -Is)" | tee -a "$LOG"
  /home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_lowkv_exploration_20260712.py >> "$LOG" 2>&1 || true
  if [[ "$complete" -ge "${#expected[@]}" ]]; then
    break
  fi
  sleep 300
done

/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_lowkv_exploration_20260712.py | tee -a "$LOG"
echo "DONE lowkv exploration summary $(date -Is)" | tee -a "$LOG"
