#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

REPORT_DIR="outputs/riskkv_v19_v314_v315_bm25_bridge_smoke_20260711"
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
      for version in v314_b16_windowvote_bm25bridge v315_b128_bm25bridge; do
        local dir="outputs/riskkv_v19_${version}_${task}_20260711_bm25_bridge_smoke_m20_bDyn_pDyn"
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
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_narrativeqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_qasper_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_multifieldqa_en_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_hotpotqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_2wikimqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v314_b16_windowvote_bm25bridge_musique_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_narrativeqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_qasper_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_multifieldqa_en_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_hotpotqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_2wikimqa_20260711_bm25_bridge_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v315_b128_bm25bridge_musique_20260711_bm25_bridge_smoke_m20_bDyn_pDyn

cat "$REPORT_DIR/summary_table.csv"
echo "DONE BM25 bridge smoke combine $(date -Is)"
