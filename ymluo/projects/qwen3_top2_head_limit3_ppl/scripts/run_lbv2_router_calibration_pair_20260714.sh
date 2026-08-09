#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 GPU SHARD DOMAINS" >&2
  exit 2
fi

GPU="$1"
SHARD="$2"
DOMAINS="$3"

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

run_rung() {
  local label="$1"
  local methods="$2"
  local policy="$3"
  env \
    GPUS="$GPU" \
    METHOD="$methods" \
    POLICY="$policy" \
    LABEL="${label}_${SHARD}" \
    SAMPLES=89 \
    MAX_CONTEXT_TOKENS=32000 \
    DOMAINS="$DOMAINS" \
    DATA_JSON="datasets/lbv2_frozen_splits_20260714/router_calibration.json" \
    CHOICE_DECODE=1 \
    SPARSE_QUERY_PHYSICAL_MASK=1 \
    SPARSE_POSITION_MODE=original \
    STAMP="20260714_lbv2_router_calibration" \
    LOCK_ROOT="/tmp/riskkv_lbv2_router_calibration_20260714" \
    bash scripts/run_longbench_v2_operator_eval_20260714.sh
}

run_rung basefull "ours_page_gather,full_kv" configs/riskkv_operator_contract_v468_lpcm_choice_20260714.json
run_rung b2048 "ours_page_gather" configs/riskkv_operator_contract_v469_lpcm_choice_b2048_20260714.json
