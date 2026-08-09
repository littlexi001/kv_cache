#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

QA_TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique"
BASE_V300="outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"
REPORT_DIR="outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711"
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
      for version in v301_b16_group4_highrecall v302_b16_group2_highrecall; do
        local dir="outputs/riskkv_v19_${version}_${task}_20260711_b16_group_sweep_m100_bDyn_pDyn"
        if ! need_file "$dir/task_results.csv"; then
          echo "WAIT missing $dir/task_results.csv $(date -Is)"
          missing=1
        fi
      done
    done
    for task in narrativeqa qasper multifieldqa_en 2wikimqa musique; do
      local old_dir="outputs/riskkv_v19_v289_b16_highrecall_${task}_20260711_b16_m100_m100_bDyn_pDyn"
      if ! need_file "$old_dir/task_results.csv"; then
        echo "WAIT missing $old_dir/task_results.csv $(date -Is)"
        missing=1
      fi
    done
    local hotpot_dir="outputs/riskkv_v19_v289_b16_highrecall_hotpotqa_20260711_b16_group_sweep_m100_bDyn_pDyn"
    if ! need_file "$hotpot_dir/task_results.csv"; then
      echo "WAIT missing $hotpot_dir/task_results.csv $(date -Is)"
      missing=1
    fi
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
    --replace_tasks "$QA_TASKS" \
    "$@"
}

wait_for_outputs

combine_variant \
  "outputs/riskkv_v19_v301_b16_group4_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn" \
  outputs/riskkv_v19_v301_b16_group4_highrecall_narrativeqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v301_b16_group4_highrecall_qasper_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v301_b16_group4_highrecall_multifieldqa_en_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v301_b16_group4_highrecall_hotpotqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v301_b16_group4_highrecall_2wikimqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v301_b16_group4_highrecall_musique_20260711_b16_group_sweep_m100_bDyn_pDyn

combine_variant \
  "outputs/riskkv_v19_v302_b16_group2_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn" \
  outputs/riskkv_v19_v302_b16_group2_highrecall_narrativeqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v302_b16_group2_highrecall_qasper_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v302_b16_group2_highrecall_multifieldqa_en_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v302_b16_group2_highrecall_hotpotqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v302_b16_group2_highrecall_2wikimqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v302_b16_group2_highrecall_musique_20260711_b16_group_sweep_m100_bDyn_pDyn

combine_variant \
  "outputs/riskkv_v19_v303_v289_b16_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn" \
  outputs/riskkv_v19_v289_b16_highrecall_narrativeqa_20260711_b16_m100_m100_bDyn_pDyn \
  outputs/riskkv_v19_v289_b16_highrecall_qasper_20260711_b16_m100_m100_bDyn_pDyn \
  outputs/riskkv_v19_v289_b16_highrecall_multifieldqa_en_20260711_b16_m100_m100_bDyn_pDyn \
  outputs/riskkv_v19_v289_b16_highrecall_hotpotqa_20260711_b16_group_sweep_m100_bDyn_pDyn \
  outputs/riskkv_v19_v289_b16_highrecall_2wikimqa_20260711_b16_m100_m100_bDyn_pDyn \
  outputs/riskkv_v19_v289_b16_highrecall_musique_20260711_b16_m100_m100_bDyn_pDyn

"$PYTHON" - <<'PY' > "$REPORT_DIR/summary_table.csv"
import csv
from pathlib import Path

rows = []
for name, directory in [
    ("v300_main", "outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn"),
    ("v301_b16_group4", "outputs/riskkv_v19_v301_b16_group4_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn"),
    ("v302_b16_group2", "outputs/riskkv_v19_v302_b16_group2_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn"),
    ("v303_v289_b16_group8", "outputs/riskkv_v19_v303_v289_b16_highrecall_with_v300_nonqa_20260711_m100_bDyn_pDyn"),
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
echo "DONE b16 group sweep combine $(date -Is)"
