#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

BASE="outputs/riskkv_v19_v461_schema_operator_contract_m20_combined_20260713"
OVERLAY="outputs/riskkv_v19_v462_contract_length_calibrated_v462_m20_20260713_m20_bDyn_pDyn"
OUT="outputs/riskkv_v19_v462_operator_contract_m20_combined_20260713"
FULL="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv"

while [[ ! -f "$BASE/task_results.csv" || ! -f "$OVERLAY/task_results.csv" ]]; do
  echo "WAIT_V462_INPUTS $(date -Is)"
  sleep 30
done

python scripts/compose_task_result_overlays_20260713.py \
  --output-dir "$OUT" \
  --base "$BASE" \
  --overlay "$OVERLAY"

python scripts/summarize_operator_contract_v461_20260713.py \
  --ours "$OUT/task_results.csv" \
  --full "$FULL" \
  --output-dir "$OUT" \
  > "$OUT/matched_summary.log" 2>&1

cat "$OUT/matched_summary.log"
