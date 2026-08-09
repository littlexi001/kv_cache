#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

BASE_V300="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
REPORT_DIR="outputs/riskkv_v19_v324_qasper_narrative_nofull_m100_20260711"
mkdir -p "$REPORT_DIR"

while true; do
  missing=0
  for task in narrativeqa qasper; do
    dir="outputs/riskkv_v19_v324_qasper_narrative_nofull_${task}_20260711_v324_qasper_narrative_nofull_m100_m100_bDyn_pDyn"
    if [[ ! -f "$dir/task_results.csv" ]]; then
      echo "WAIT missing $dir/task_results.csv $(date -Is)"
      missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] && break
  sleep 180
done

"$PYTHON" scripts/combine_replace_tasks_20260711.py \
  --base_dir "$BASE_V300" \
  --output_dir "outputs/riskkv_v19_v324_qasper_narrative_nofull_with_v300_other_20260711_m100_bDyn_pDyn" \
  --replace_tasks "narrativeqa,qasper" \
  outputs/riskkv_v19_v324_qasper_narrative_nofull_narrativeqa_20260711_v324_qasper_narrative_nofull_m100_m100_bDyn_pDyn \
  outputs/riskkv_v19_v324_qasper_narrative_nofull_qasper_20260711_v324_qasper_narrative_nofull_m100_m100_bDyn_pDyn

"$PYTHON" - <<'PY' > "$REPORT_DIR/summary_table.csv"
import csv
from pathlib import Path

rows = []
for name, directory in [
    ("v300_main", "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"),
    ("v324_qasper_narrative_nofull", "outputs/riskkv_v19_v324_qasper_narrative_nofull_with_v300_other_20260711_m100_bDyn_pDyn"),
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
echo "DONE v324 qasper+narrative no-full combine $(date -Is)"
