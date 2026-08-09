#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

BASE_V300="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
V329_TASK_DIR="outputs/riskkv_v19_v329_2wiki_b128_bm25bridge_20260711_v329_2wiki_b128_bm25bridge_m100_m100_bDyn_pDyn"
V330_TASK_DIR="outputs/riskkv_v19_v330_multifield_shortdecode_20260711_v330_multifield_shortdecode_m100_m100_bDyn_pDyn"
COMBINED_DIR="outputs/riskkv_v19_v331_2wiki_bm25_multifield_shortdecode_with_v300_other_20260711_m100_bDyn_pDyn"
REPORT_DIR="outputs/riskkv_v19_v331_2wiki_bm25_multifield_shortdecode_m100_20260711"
mkdir -p "$REPORT_DIR"

for task_dir in "$V329_TASK_DIR" "$V330_TASK_DIR"; do
  while [[ ! -f "$task_dir/task_results.csv" ]]; do
    echo "WAIT missing $task_dir/task_results.csv $(date -Is)"
    sleep 180
  done
done

"$PYTHON" scripts/combine_replace_tasks_20260711.py \
  --base_dir "$BASE_V300" \
  --output_dir "$COMBINED_DIR" \
  --replace_tasks "2wikimqa,multifieldqa_en" \
  "$V329_TASK_DIR" \
  "$V330_TASK_DIR"

"$PYTHON" - <<'PY' > "$REPORT_DIR/summary_table.csv"
import csv
from pathlib import Path

rows = []
for name, directory in [
    ("v300_main", "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"),
    ("v329_2wiki_b128_bm25bridge", "outputs/riskkv_v19_v329_2wiki_b128_bm25bridge_with_v300_other_20260711_m100_bDyn_pDyn"),
    ("v330_multifield_shortdecode", "outputs/riskkv_v19_v330_multifield_shortdecode_with_v300_other_20260711_m100_bDyn_pDyn"),
    ("v331_combined", "outputs/riskkv_v19_v331_2wiki_bm25_multifield_shortdecode_with_v300_other_20260711_m100_bDyn_pDyn"),
]:
    path = Path(directory) / "summary.csv"
    if not path.exists():
        continue
    with path.open(newline="", encoding="utf-8") as handle:
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
echo "DONE v331 combined v329 v330 $(date -Is)"
