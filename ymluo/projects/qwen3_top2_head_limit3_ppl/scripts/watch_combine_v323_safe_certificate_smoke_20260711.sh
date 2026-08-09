#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

REPORT_DIR="outputs/riskkv_v19_v323_safe_certificate_smoke_20260711"
FULL_DIR="outputs/riskkv_fullkv_m100_same_samples_20260710"
V300_DIR="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
mkdir -p "$REPORT_DIR"

while true; do
  missing=0
  for task in narrativeqa musique; do
    dir="outputs/riskkv_v19_v323_safe_certificate_${task}_20260711_v323_safe_certificate_smoke_m20_bDyn_pDyn"
    if [[ ! -f "$dir/task_results.csv" ]]; then
      echo "WAIT missing $dir/task_results.csv $(date -Is)"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] && break
  sleep 120
done

"$PYTHON" scripts/compare_smoke_to_baselines_20260711.py \
  --full_dir "$FULL_DIR" \
  --v300_dir "$V300_DIR" \
  --output_dir "$REPORT_DIR" \
  outputs/riskkv_v19_v323_safe_certificate_narrativeqa_20260711_v323_safe_certificate_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v323_safe_certificate_musique_20260711_v323_safe_certificate_smoke_m20_bDyn_pDyn

cat "$REPORT_DIR/summary_table.csv"
echo "DONE v323 safe-certificate smoke combine $(date -Is)"
