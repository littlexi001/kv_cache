#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

REPORT_DIR="outputs/riskkv_v19_v316_v317_qa_shortdecode_smoke_20260711"
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
      for version in v316_v300_shortdecode_balanced v317_v300_shortdecode_aggressive; do
        local dir="outputs/riskkv_v19_${version}_${task}_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn"
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
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_narrativeqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_qasper_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_multifieldqa_en_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_hotpotqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_2wikimqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v316_v300_shortdecode_balanced_musique_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_narrativeqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_qasper_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_multifieldqa_en_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_hotpotqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_2wikimqa_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn \
  outputs/riskkv_v19_v317_v300_shortdecode_aggressive_musique_20260711_qa_shortdecode_smoke_m20_bDyn_pDyn

cat "$REPORT_DIR/summary_table.csv"
echo "DONE QA short-decode smoke combine $(date -Is)"
