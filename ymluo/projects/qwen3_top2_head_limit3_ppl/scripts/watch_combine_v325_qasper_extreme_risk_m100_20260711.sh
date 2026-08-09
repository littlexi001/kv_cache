#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

BASE_V300="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
REPORT_DIR="outputs/riskkv_v19_v325_qasper_extreme_risk_m100_20260711"
mkdir -p "$REPORT_DIR"

QASPER_DIR="outputs/riskkv_v19_v325_qasper_extreme_risk_qasper_20260711_v325_qasper_extreme_risk_m100_m100_bDyn_pDyn"
while [[ ! -f "$QASPER_DIR/task_results.csv" ]]; do
  echo "WAIT missing $QASPER_DIR/task_results.csv $(date -Is)"
  sleep 180
done

"$PYTHON" scripts/combine_replace_tasks_20260711.py \
  --base_dir "$BASE_V300" \
  --output_dir "outputs/riskkv_v19_v325_qasper_extreme_risk_with_v300_other_20260711_m100_bDyn_pDyn" \
  --replace_tasks "qasper" \
  "$QASPER_DIR"

"$PYTHON" - <<'PY' > "$REPORT_DIR/summary_table.csv"
import csv
from pathlib import Path

rows = []
for name, directory in [
    ("v300_main", "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"),
    ("v325_qasper_extreme_risk", "outputs/riskkv_v19_v325_qasper_extreme_risk_with_v300_other_20260711_m100_bDyn_pDyn"),
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
echo "DONE v325 qasper extreme-risk combine $(date -Is)"
