#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

STAMP="${STAMP:-20260711_b16_compressed_mid_smoke}"
SAMPLES="${SAMPLES:-20}"
REPORT_DIR="outputs/riskkv_v19_v327_v328_b16_compressed_mid_smoke_20260711"
FULL_DIR="outputs/riskkv_fullkv_m100_same_samples_20260710"
V300_DIR="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
mkdir -p "$REPORT_DIR"

need_file() {
  local file="$1"
  [[ -f "$file" ]]
}

wait_for_outputs() {
  local missing=0
  while true; do
    missing=0
    for task in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique; do
      for version in v327_b16_compressed_mid v328_b16_window64_compressed_mid; do
        local dir="outputs/riskkv_v19_${version}_${task}_${STAMP}_m${SAMPLES}_bDyn_pDyn"
        if ! need_file "$dir/task_results.csv"; then
          echo "WAIT missing $dir/task_results.csv $(date -Is)"
          missing=1
        fi
      done
    done
    if [[ "$missing" -eq 0 ]]; then
      break
    fi
    sleep 120
  done
}

wait_for_outputs

"$PYTHON" scripts/compare_smoke_to_baselines_20260711.py \
  --full_dir "$FULL_DIR" \
  --v300_dir "$V300_DIR" \
  --output_dir "$REPORT_DIR" \
  outputs/riskkv_v19_v327_b16_compressed_mid_narrativeqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v327_b16_compressed_mid_qasper_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v327_b16_compressed_mid_multifieldqa_en_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v327_b16_compressed_mid_hotpotqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v327_b16_compressed_mid_2wikimqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v327_b16_compressed_mid_musique_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_narrativeqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_qasper_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_multifieldqa_en_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_hotpotqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_2wikimqa_${STAMP}_m${SAMPLES}_bDyn_pDyn \
  outputs/riskkv_v19_v328_b16_window64_compressed_mid_musique_${STAMP}_m${SAMPLES}_bDyn_pDyn

cat "$REPORT_DIR/summary_table.csv"
echo "DONE B16 compressed-mid smoke combine $(date -Is)"
