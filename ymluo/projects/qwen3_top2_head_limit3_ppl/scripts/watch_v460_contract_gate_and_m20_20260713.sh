#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

PYTHON="/home/fdong/miniconda3/envs/moe/bin/python"
SUMMARY="outputs/riskkv_v19_v460_operator_contract_router_20260713/router_summary.csv"
CONFIG="configs/riskkv_operator_contract_router_v460_20260713.json"
RUNNER="scripts/run_riskkv_task_policy_v19_one_20260709.sh"
OUT="outputs/riskkv_v19_v460_generic_operator_contract_v460_m20_20260713_m20_bDyn_pDyn"

while [[ ! -s "$SUMMARY" || ! -s "$CONFIG" ]]; do
  echo "WAIT v460 contract training $(date -Is)"
  sleep 60
done

"$PYTHON" - <<'PY'
import csv
import sys

rows = list(csv.DictReader(open("outputs/riskkv_v19_v460_operator_contract_router_20260713/router_summary.csv")))
sample = next(row for row in rows if row["split"] == "test")
loto = [row for row in rows if row["split"] == "loto"]
sample_acc = float(sample["accuracy"])
mean_loto = sum(float(row["accuracy"]) for row in loto) / max(1, len(loto))
min_loto = min(float(row["accuracy"]) for row in loto)
print(f"v460 contract sample_acc={sample_acc:.4f} mean_loto={mean_loto:.4f} min_loto={min_loto:.4f}")
if sample_acc < 0.90 or mean_loto < 0.75 or min_loto < 0.50:
    print("v460 contract gate failed; skip GPU generation.", file=sys.stderr)
    sys.exit(2)
PY

GPUS="0,1,2,4,5,6,7" SAMPLES=20 LABEL="v460_generic_operator_contract" \
  STAMP="v460_m20_20260713" POLICY="$CONFIG" bash "$RUNNER"

"$PYTHON" - <<'PY'
import csv
import hashlib
import json
from collections import Counter

CONTRACT = {
    "narrativeqa": "retrieve", "qasper": "retrieve", "multifieldqa_en": "retrieve",
    "hotpotqa": "retrieve", "2wikimqa": "retrieve", "musique": "retrieve", "triviaqa": "retrieve",
    "gov_report": "aggregate", "qmsum": "aggregate", "multi_news": "aggregate", "samsum": "aggregate",
    "trec": "structured", "passage_count": "structured", "passage_retrieval_en": "structured",
    "lcc": "code", "repobench-p": "code",
}

def fold(task, sample_id):
    digest = hashlib.md5(f"{task}\t{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 5

def f(row, key):
    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0

path = "outputs/riskkv_v19_v460_generic_operator_contract_v460_m20_20260713_m20_bDyn_pDyn/task_results.csv"
full_path = "outputs/riskkv_fullkv_m100_same_samples_20260710/task_results.csv"
rows = list(csv.DictReader(open(path)))
full = {(row["task"], row["sample_id"]): row for row in csv.DictReader(open(full_path))}
for split, subset in [
    ("all", rows),
    ("fold0", [row for row in rows if fold(row["task"], row["sample_id"]) == 0]),
]:
    pairs = [(row, full.get((row["task"], row["sample_id"]))) for row in subset]
    pairs = [(row, baseline) for row, baseline in pairs if baseline is not None]
    score = sum(f(row, "score") for row, _ in pairs) / max(1, len(pairs))
    full_score = sum(f(base, "score") for _, base in pairs) / max(1, len(pairs))
    kv = sum(f(row, "keep_fraction") for row, _ in pairs) / max(1, len(pairs))
    total = sum(f(row, "total_seconds") for row, _ in pairs) / max(1, len(pairs))
    full_total = sum(f(base, "total_seconds") for _, base in pairs) / max(1, len(pairs))
    route_acc = sum(row.get("ours_operator_mode") == CONTRACT[row["task"]] for row, _ in pairs) / max(1, len(pairs))
    modes = Counter(row.get("ours_operator_mode", "") for row, _ in pairs)
    print(json.dumps({
        "split": split,
        "samples": len(pairs),
        "score": score,
        "full_score": full_score,
        "score_vs_full": score / full_score if full_score else None,
        "kv": kv,
        "total_speed_vs_full": full_total / total if total else None,
        "route_accuracy": route_acc,
        "operator_counts": dict(modes),
    }, ensure_ascii=False))
PY
