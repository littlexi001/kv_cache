#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

G1="outputs/riskkv_v19_v466_m100_g1_v466_m100_20260713_m100_bDyn_pDyn"
G2="outputs/riskkv_v19_v466_m100_g2_v466_m100_20260713_m100_bDyn_pDyn"
G3="outputs/riskkv_v19_v466_m100_g3_v466_m100_20260713_m100_bDyn_pDyn"
OUT="outputs/riskkv_v19_v466_operator_contract_m100_combined_20260713"
FULL="outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv"

python scripts/combine_split_task_results_20260711.py \
  --wait \
  --wait_interval_seconds 60 \
  --output_dir "$OUT" \
  "$G1" "$G2" "$G3"

python scripts/summarize_operator_contract_v461_20260713.py \
  --ours "$OUT/task_results.csv" \
  --full "$FULL" \
  --output-dir "$OUT" \
  > "$OUT/matched_summary.log" 2>&1

cat "$OUT/matched_summary.log"
