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
  local policy="$2"
  env \
    GPUS="$GPU" \
    METHOD="ours_page_gather" \
    POLICY="$policy" \
    LABEL="${label}_${SHARD}" \
    SAMPLES=222 \
    MAX_CONTEXT_TOKENS=32000 \
    DOMAINS="$DOMAINS" \
    DATA_JSON="datasets/lbv2_frozen_splits_20260714/router_train.json" \
    CHOICE_DECODE=1 \
    SPARSE_QUERY_PHYSICAL_MASK=1 \
    SPARSE_POSITION_MODE=original \
    STAMP="20260714_lbv2_router_train" \
    LOCK_ROOT="/tmp/riskkv_lbv2_router_train_20260714" \
    bash scripts/run_longbench_v2_operator_eval_20260714.sh
}

run_rung base configs/riskkv_operator_contract_v468_lpcm_choice_20260714.json
run_rung b2048 configs/riskkv_operator_contract_v469_lpcm_choice_b2048_20260714.json
