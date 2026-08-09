#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

RUNNER="scripts/run_riskkv_task_policy_v19_one_20260709.sh"
TASKS="gov_report,qmsum,multi_news,trec,samsum,passage_count,passage_retrieval_en"
SAMPLES=10
STAMP="20260712_multichannel"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

# label|budget|page|mode
ACTIONS=(
  "spread_b384_p16|384|16|spread"
  "spread_b384_p64|384|64|spread"
  "structure_b384_p16|384|16|structure"
  "tri_b384_p16|384|16|tri"
  "tri_b512_p16|512|16|tri"
  "tri_b512_p64|512|64|tri"
)

policy_json() {
  local budget="$1"
  local page="$2"
  local mode="$3"
  local common
  common="\"budget_tokens\":${budget},\"page_tokens\":${page},\"sink_tokens\":32,\"recent_tokens\":32,\"short_decode\":false,\"direct_structured_answer\":false"
  if [[ "$mode" == "spread" ]]; then
    echo "{\"__runtime_constraints\":{\"direct_structured_answer\":false},\"*\":{${common},\"scorer\":\"hybrid_late_mmr_multiscale_idf_spread_flow\",\"coverage_weight\":0.30,\"ours_idf_mix\":0.70,\"ours_spread_budget_fraction\":0.34,\"ours_spread_gap_threshold\":1.0,\"ours_spread_bins\":6,\"ours_spread_min_score\":0.0}}"
  elif [[ "$mode" == "structure" ]]; then
    echo "{\"__runtime_constraints\":{\"direct_structured_answer\":false},\"*\":{${common},\"scorer\":\"hybrid_late_mmr_multiscale_flow\",\"structured_fingerprint\":true,\"structured_fingerprint_budget_fraction\":0.28,\"coverage_certificate\":true,\"coverage_certificate_budget_fraction\":0.24,\"coverage_certificate_min_terms\":1,\"passage_closure\":true,\"passage_closure_budget_fraction\":0.14,\"passage_closure_radius_pages\":1}}"
  else
    echo "{\"__runtime_constraints\":{\"direct_structured_answer\":false},\"*\":{${common},\"scorer\":\"hybrid_late_mmr_multiscale_idf_spread_flow\",\"coverage_weight\":0.24,\"ours_idf_mix\":0.70,\"ours_spread_budget_fraction\":0.26,\"ours_spread_gap_threshold\":1.0,\"ours_spread_bins\":6,\"ours_spread_min_score\":0.0,\"structured_fingerprint\":true,\"structured_fingerprint_budget_fraction\":0.20,\"coverage_certificate\":true,\"coverage_certificate_budget_fraction\":0.18,\"coverage_certificate_min_terms\":1,\"passage_closure\":true,\"passage_closure_budget_fraction\":0.10,\"passage_closure_radius_pages\":1}}"
  fi
}

run_action() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r label budget page mode <<< "$spec"
  local policy
  policy="$(policy_json "$budget" "$page" "$mode")"
  local log="$LOG_ROOT/nohup_v444_${label}_m10_20260712.log"
  echo "START v444 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
  GPUS="$gpu" SAMPLES="$SAMPLES" TASKS="$TASKS" LABEL="v444_${label}" \
    STAMP="$STAMP" POLICY="$policy" bash "$RUNNER" >> "$log" 2>&1
  echo "DONE v444 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
}

worker() {
  local worker_id="$1"
  local gpu="$2"
  local index
  for index in "${!ACTIONS[@]}"; do
    if (( index % 3 == worker_id )); then
      run_action "$gpu" "${ACTIONS[$index]}"
    fi
  done
}

worker 0 5 &
pid0=$!
worker 1 6 &
pid1=$!
worker 2 7 &
pid2=$!
wait "$pid0"
wait "$pid1"
wait "$pid2"
echo "V444 Pure multichannel M10 complete $(date -Is)"
