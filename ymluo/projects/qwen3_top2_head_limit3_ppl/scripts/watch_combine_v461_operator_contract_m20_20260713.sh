#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

G1="outputs/riskkv_v19_v461_schema_contract_g1_v461_m20_20260713_m20_bDyn_pDyn"
G2="outputs/riskkv_v19_v461_schema_contract_g2_v461_m20_20260713_m20_bDyn_pDyn"
G3="outputs/riskkv_v19_v461_schema_contract_g3_v461_m20_20260713_m20_bDyn_pDyn"
OUT="outputs/riskkv_v19_v461_schema_operator_contract_m20_combined_20260713"
FULL="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv"

python scripts/combine_split_task_results_20260711.py \
  --wait \
  --wait_interval_seconds 30 \
  --output_dir "$OUT" \
  "$G1" "$G2" "$G3"

python scripts/summarize_operator_contract_v461_20260713.py \
  --ours "$OUT/task_results.csv" \
  --full "$FULL" \
  --output-dir "$OUT" \
  > "$OUT/matched_summary.log" 2>&1

cat "$OUT/matched_summary.log"
