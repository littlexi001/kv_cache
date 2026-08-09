#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

REPORT_DIR="outputs/riskkv_v19_v318_v319_qasper_bm25_budget_smoke_20260711"
FULL_DIR="outputs/riskkv_fullkv_m100_same_samples_20260710"
V300_DIR="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
mkdir -p "$REPORT_DIR"

wait_for_outputs() {
  while true; do
    missing=0
    for version in v318_qasper_b128_bm25bridge_1280 v319_qasper_b128_bm25bridge_1024; do
      local dir="outputs/riskkv_v19_${version}_20260711_qasper_bm25_budget_smoke_m20_bDyn_pDyn"
      if [[ ! -f "$dir/task_results.csv" ]]; then
        echo "WAIT missing $dir/task_results.csv $(date -Is)"
        missing=1
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      break
    fi
    sleep 90
  done
}

wait_for_outputs

"$PYTHON" scripts/compare_smoke_to_baselines_20260711.py \
  --full_dir "$FULL_DIR" \
  --v300_dir "$V300_DIR" \
  --output_dir "$REPORT_DIR" \
  outputs/riskkv_v19_v318_qasper_b128_bm25bridge_1280_20260711_qasper_bm25_budget_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v319_qasper_b128_bm25bridge_1024_20260711_qasper_bm25_budget_smoke_m20_bDyn_pDyn

cat "$REPORT_DIR/detail_table.csv"
echo "DONE qasper BM25 budget smoke combine $(date -Is)"
