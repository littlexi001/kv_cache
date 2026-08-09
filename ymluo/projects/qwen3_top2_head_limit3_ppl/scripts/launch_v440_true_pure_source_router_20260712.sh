#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

POLICY="configs/riskkv_task_policy_v440_true_pure_source_router_20260712.json"
RUNNER="scripts/run_riskkv_task_policy_v19_one_20260709.sh"
PYTHON="/home/fdong/miniconda3/envs/moe/bin/python"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

run_one() {
  local samples="$1"
  local label="$2"
  local stamp="$3"
  local log="$LOG_ROOT/nohup_${label}_${stamp}.log"
  echo "LAUNCH ${label} samples=${samples} policy=${POLICY} $(date -Is)" | tee -a "$log"
  SAMPLES="$samples" LABEL="$label" STAMP="$stamp" POLICY="$POLICY" \
    bash "$RUNNER" >> "$log" 2>&1
}

run_one 20 "v440_true_pure_source_router" "v440_m20_20260712"

"$PYTHON" - <<'PY'
import csv
import sys
from pathlib import Path

out = Path("outputs/riskkv_v19_v440_true_pure_source_router_v440_m20_20260712_m20_bDyn_pDyn/task_results.csv")
rows = list(csv.DictReader(out.open())) if out.exists() else []
if not rows:
    print(f"M20 result missing or empty: {out}", file=sys.stderr)
    sys.exit(1)
score = sum(float(row["score"]) for row in rows) / len(rows)
kv = sum(float(row["keep_fraction"]) for row in rows) / len(rows)
online = sum(float(row["online_seconds"]) for row in rows) / len(rows)
direct = sum(int(float(row.get("ours_direct_structured_answer_used", 0) or 0)) for row in rows)
print(f"M20 score={score:.4f} kv={kv:.4%} online={online:.4f}s direct_used={direct}")
if direct != 0:
    print("True-Pure gate failed: direct structured answer was used.", file=sys.stderr)
    sys.exit(3)
if score < 0.30 or kv > 0.12:
    print("M20 quality/KV gate failed; skip M100.", file=sys.stderr)
    sys.exit(2)
PY

run_one 100 "v440_true_pure_source_router" "v440_m100_20260712"
