#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

mkdir -p outputs/logs

watch_one() {
  local label="$1"
  local stamp="$2"
  local policy="$3"
  local min_speed="$4"
  local log="outputs/logs/watch_${label}_${stamp}.log"
  {
    echo "WAIT_CONFIG label=${label} policy=${policy} $(date -Is)"
    while [[ ! -f "$policy" ]]; do
      sleep 60
    done
    echo "FOUND_CONFIG label=${label} policy=${policy} $(date -Is)"
    env \
      LABEL="$label" \
      STAMP="$stamp" \
      POLICY="$policy" \
      MAX_KV=0.055 \
      MIN_VS_FULL=0.95 \
      MIN_SPEED="$min_speed" \
      GPUS="${GPUS:-6,7,2,4,0,1,3,5}" \
      bash scripts/watch_fixed_policy_gate_20260712.sh
  } > "$log" 2>&1
}

watch_one \
  "v424_latency_frontier_router060" \
  "20260712_latency_v424" \
  "configs/riskkv_task_policy_v424_latency_frontier_router060_20260712.json" \
  "6.0" &

watch_one \
  "v425_latency_frontier_router068" \
  "20260712_latency_v425" \
  "configs/riskkv_task_policy_v425_latency_frontier_router068_20260712.json" \
  "6.8" &

wait
