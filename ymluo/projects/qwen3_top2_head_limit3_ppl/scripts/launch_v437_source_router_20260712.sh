#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

POLICY="configs/riskkv_task_policy_v437_source_router_v428_20260712.json"
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

if [[ "${SKIP_M20:-0}" != "1" ]]; then
  run_one 20 "v437_source_router_v428" "v437_m20_20260712"
fi

"$PYTHON" - <<'PY'
import csv
import sys
from pathlib import Path

out = Path("outputs/riskkv_v19_v437_source_router_v428_v437_m20_20260712_m20_bDyn_pDyn/task_results.csv")
if not out.exists():
    print(f"M20 result missing: {out}", file=sys.stderr)
    sys.exit(1)
rows = list(csv.DictReader(out.open()))
score = sum(float(r["score"]) for r in rows) / max(1, len(rows))
kv = sum(float(r["keep_fraction"]) for r in rows) / max(1, len(rows))
online = sum(float(r["online_seconds"]) for r in rows) / max(1, len(rows))
print(f"M20 score={score:.4f} kv={kv:.4%} online={online:.4f}s")
if score < 0.30 or kv > 0.12:
    print("M20 gate failed; skip M100.", file=sys.stderr)
    sys.exit(2)
PY

run_one 100 "v437_source_router_v428" "v437_m100_20260712"
