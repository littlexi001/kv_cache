#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

STAMP="${STAMP:-20260704_budget_sweep_v11}"
SAMPLES="${SAMPLES:-5}"
BUDGETS="${BUDGETS:-256 512 1024 2048}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-float16}"
SUITE="${SUITE:-both}" # longbench, ruler, both
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-4096}"
LONG_METHODS="${LONG_METHODS:-FullKV StreamingLLM H2O SnapKV PyramidKV AdaKV}"
RULER_METHODS="${RULER_METHODS:-FullKV StreamingLLM H2O SnapKV PyramidKV}"
LONG_TASKS="${LONG_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,gov_report,multi_news}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/bin/python}"

LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

summarize_longbench() {
  local out_root="outputs/kvcache_factory_official_longbench_${SAMPLES}shot_${STAMP}"
  if [[ ! -d "$out_root" ]]; then
    return
  fi
  source /home/fdong/miniconda3/bin/activate moe
  python src/summarize_kvcache_factory_longbench.py \
    --input_dir "$out_root" \
    --output_csv "$out_root/summary.csv" \
    --output_json "$out_root/summary.json"
  python - "$STAMP" "$SAMPLES" "$out_root" <<'PY'
import csv
import pathlib
import re
import sys

stamp, samples, out_root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
rows = []
for path in sorted(pathlib.Path("outputs/logs").glob(f"kvcache_factory_longbench_*_b*_{samples}shot_{stamp}.log")):
    name = path.name.removeprefix("kvcache_factory_longbench_")
    method, rest = name.split("_b", 1)
    budget = rest.split("_", 1)[0]
    current = None
    seen = set()
    for line in path.read_text(errors="ignore").splitlines():
        match = re.search(r"Working on max_capacity_prompts \d+ dataset ([^ ]+) -", line)
        if match:
            current = match.group(1)
            continue
        if current and "100%" in line and "s/it" in line:
            match_time = re.search(r",\s*([0-9.]+)s/it\]", line)
            if match_time:
                key = (budget, method, current, len([row for row in rows if row["budget"] == budget and row["method"] == method and row["task"] == current]))
                if key not in seen:
                    rows.append({"budget": budget, "method": method, "task": current, "eval_seconds": float(match_time.group(1))})
                    seen.add(key)
                current = None
with (out_root / "timing_from_logs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["budget", "method", "task", "eval_seconds"])
    writer.writeheader()
    writer.writerows(rows)
PY
}

summarize_ruler() {
  local out_root="outputs/kvcache_factory_official_ruler_${SAMPLES}shot_${STAMP}"
  if [[ ! -d "$out_root" ]]; then
    return
  fi
  "$PYTHON" src/summarize_kvcache_factory_ruler.py \
    --input_dir "$out_root" \
    --output_csv "$out_root/summary.csv" \
    --output_json "$out_root/summary.json"
  python - "$STAMP" "$SAMPLES" "$out_root" <<'PY'
import csv
import pathlib
import re
import sys

stamp, samples, out_root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
rows = []
for path in sorted(pathlib.Path("outputs/logs").glob(f"kvcache_factory_ruler_*_b*_{samples}shot_{stamp}.log")):
    name = path.name.removeprefix("kvcache_factory_ruler_")
    method, rest = name.split("_b", 1)
    budget = rest.split("_", 1)[0]
    current = None
    for line in path.read_text(errors="ignore").splitlines():
        match = re.search(r"dataset: ([^ ]+) -", line)
        if match:
            current = match.group(1)
            continue
        if current and "100%" in line and "s/it" in line:
            match_time = re.search(r",\s*([0-9.]+)s/it\]", line)
            if match_time:
                rows.append({"budget": budget, "method": method, "task": current, "eval_seconds": float(match_time.group(1))})
                current = None
with (out_root / "timing_from_logs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["budget", "method", "task", "eval_seconds"])
    writer.writeheader()
    writer.writerows(rows)
PY
}

case "$SUITE" in
  longbench|both)
    STAMP="$STAMP" \
    MODEL="$MODEL" \
    SAMPLES="$SAMPLES" \
    BUDGETS="$BUDGETS" \
    METHODS="$LONG_METHODS" \
    TASKS="$LONG_TASKS" \
    ATTN="$ATTN" \
    DTYPE="$DTYPE" \
    CONTINUE_ON_ERROR=1 \
      bash scripts/run_kvcache_factory_official_longbench_sweep_server.sh
    summarize_longbench
    ;;
esac

case "$SUITE" in
  ruler|both)
    STAMP="$STAMP" \
    MODEL="$MODEL" \
    SAMPLES="$SAMPLES" \
    BUDGETS="$BUDGETS" \
    METHODS="$RULER_METHODS" \
    CONTEXT_LENGTHS="$CONTEXT_LENGTHS" \
    ATTN="$ATTN" \
    CONTINUE_ON_ERROR=1 \
      bash scripts/run_kvcache_factory_official_ruler_sweep_server.sh
    summarize_ruler
    ;;
esac

echo "budget sweep finished: STAMP=$STAMP SAMPLES=$SAMPLES SUITE=$SUITE"
