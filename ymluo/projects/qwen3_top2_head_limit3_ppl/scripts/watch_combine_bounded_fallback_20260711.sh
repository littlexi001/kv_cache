#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

REPLACE_TASKS="narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,repobench-p"
BASE_V300="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
REPORT_DIR="outputs/riskkv_v19_v304_v305_bounded_fallback_20260711"
mkdir -p "$REPORT_DIR"

wait_for_outputs() {
  while true; do
    local missing=0
    for task in narrativeqa multifieldqa_en hotpotqa 2wikimqa musique qmsum repobench-p; do
      for version in v304_bounded4k v305_bounded3k; do
        local dir="outputs/riskkv_v19_${version}_${task}_20260711_bounded_fallback_m100_bDyn_pDyn"
        if [[ ! -f "$dir/task_results.csv" ]]; then
          echo "WAIT missing $dir/task_results.csv $(date -Is)"
          missing=1
        fi
      done
    done
    if [[ "$missing" -eq 0 ]]; then
      break
    fi
    sleep 180
  done
}

combine_variant() {
  local output_dir="$1"
  shift
  "$PYTHON" scripts/combine_replace_tasks_20260711.py \
    --base_dir "$BASE_V300" \
    --output_dir "$output_dir" \
    --replace_tasks "$REPLACE_TASKS" \
    "$@"
}

wait_for_outputs

combine_variant \
  "outputs/riskkv_v19_v304_bounded4k_with_v300_other_20260711_m100_bDyn_pDyn" \
  outputs/riskkv_v19_v304_bounded4k_narrativeqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_multifieldqa_en_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_hotpotqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_2wikimqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_musique_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_qmsum_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v304_bounded4k_repobench-p_20260711_bounded_fallback_m100_bDyn_pDyn

combine_variant \
  "outputs/riskkv_v19_v305_bounded3k_with_v300_other_20260711_m100_bDyn_pDyn" \
  outputs/riskkv_v19_v305_bounded3k_narrativeqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_multifieldqa_en_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_hotpotqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_2wikimqa_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_musique_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_qmsum_20260711_bounded_fallback_m100_bDyn_pDyn \
  outputs/riskkv_v19_v305_bounded3k_repobench-p_20260711_bounded_fallback_m100_bDyn_pDyn

"$PYTHON" - <<'PY' > "$REPORT_DIR/summary_table.csv"
import csv
from pathlib import Path

rows = []
for name, directory in [
    ("v300_main", "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"),
    ("v304_bounded4k", "outputs/riskkv_v19_v304_bounded4k_with_v300_other_20260711_m100_bDyn_pDyn"),
    ("v305_bounded3k", "outputs/riskkv_v19_v305_bounded3k_with_v300_other_20260711_m100_bDyn_pDyn"),
]:
    with (Path(directory) / "summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] == "ALL" and row["task"] == "ALL":
                rows.append({
                    "method": name,
                    "samples": row["samples"],
                    "score": row["score"],
                    "kv_keep": row["mean_keep_fraction"],
                    "online_seconds": row["mean_online_seconds"],
                    "output_dir": directory,
                })
                break

writer = csv.DictWriter(
    __import__("sys").stdout,
    fieldnames=["method", "samples", "score", "kv_keep", "online_seconds", "output_dir"],
)
writer.writeheader()
writer.writerows(rows)
PY

cat "$REPORT_DIR/summary_table.csv"
echo "DONE bounded fallback combine $(date -Is)"
