#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"

run_one() {
  local label="$1"
  local stamp="$2"
  local policy="$3"
  local max_kv="$4"
  local log="$LOG_DIR/watch_${label}_${stamp}.log"
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
      MAX_KV="$max_kv" \
      MIN_VS_FULL="${MIN_VS_FULL:-0.95}" \
      MIN_SPEED="${MIN_SPEED:-2.5}" \
      GPUS="${GPUS:-6,7,2,4,0,1,3,5}" \
      bash scripts/watch_fixed_policy_gate_20260712.sh
  } > "$log" 2>&1
}

run_one "v421_frontier_router035" "20260712_frontier_v421" "configs/riskkv_task_policy_v421_frontier_router035_20260712.json" "0.040" &
run_one "v422_frontier_router030" "20260712_frontier_v422" "configs/riskkv_task_policy_v422_frontier_router030_20260712.json" "0.035" &
run_one "v423_frontier_router040" "20260712_frontier_v423" "configs/riskkv_task_policy_v423_frontier_router040_20260712.json" "0.045" &

wait
