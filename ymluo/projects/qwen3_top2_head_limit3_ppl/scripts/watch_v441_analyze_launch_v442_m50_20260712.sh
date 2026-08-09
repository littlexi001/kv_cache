#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

PYTHON="/home/fdong/miniconda3/envs/moe/bin/python"
ANALYZER="scripts/analyze_pure_action_frontier_v441_20260712.py"
RUNNER="scripts/run_riskkv_task_policy_v19_one_20260709.sh"
TASKS="gov_report,qmsum,multi_news,trec,samsum,passage_count,passage_retrieval_en"
ANALYSIS_DIR="outputs/riskkv_v19_v441_pure_action_frontier_analysis_20260712"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

ACTIONS=(b128_p16 b256_p16 b256_p64 b512_p64 b512_p128 b1024_p128 b2048_p256)

while true; do
  missing=0
  for action in "${ACTIONS[@]}"; do
    path="outputs/riskkv_v19_v441_purefront_${action}_20260712_purefront_m10_bDyn_pDyn/task_results.csv"
    if [[ ! -s "$path" ]]; then
      missing=$((missing + 1))
    fi
  done
  if (( missing == 0 )); then
    break
  fi
  echo "WAIT v441 missing=${missing} $(date -Is)"
  sleep 120
done

"$PYTHON" "$ANALYZER"

"$PYTHON" - <<'PY'
import csv
import sys
from pathlib import Path

path = Path("outputs/riskkv_v19_v441_pure_action_frontier_analysis_20260712/frontier_summary.csv")
rows = list(csv.DictReader(path.open()))
row = next(item for item in rows if item["action"] == "oracle_best95" and item["task"] == "ALL")
ratio = float(row["score_vs_full"])
kv = float(row["kv"])
direct = sum(int(float(item.get("direct_used", 0) or 0)) for item in rows if item["action"].startswith("b"))
print(f"V441 oracle_best95 score_vs_full={ratio:.4f} kv={kv:.4%} direct_used={direct}")
if direct != 0:
    print("V441 gate failed: a Pure candidate used direct output.", file=sys.stderr)
    sys.exit(3)
if ratio < 0.95 or kv > 0.20:
    print("V441 oracle gate failed; skip M50.", file=sys.stderr)
    sys.exit(2)
PY

mapfile -t SELECTED < <("$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("outputs/riskkv_v19_v441_pure_action_frontier_analysis_20260712/selected_actions.json")
payload = json.loads(path.read_text())
actions = payload["all_actions"]
for action in payload["selected_actions_for_m50"]:
    budget = int(actions[action]["budget"])
    page = int(actions[action]["page"])
    sink = 32 if budget <= 256 else 64
    recent = 32 if budget <= 256 else (64 if budget <= 512 else 128)
    print(f"{action}|{budget}|{page}|{sink}|{recent}")
PY
)

if (( ${#SELECTED[@]} == 0 )); then
  echo "No V442 actions selected; stop."
  exit 4
fi

run_action() {
  local gpu="$1"
  local spec="$2"
  IFS='|' read -r label budget page sink recent <<< "$spec"
  local policy
  policy="{\"__runtime_constraints\":{\"direct_structured_answer\":false},\"*\":{\"budget_tokens\":${budget},\"page_tokens\":${page},\"sink_tokens\":${sink},\"recent_tokens\":${recent},\"scorer\":\"hybrid_late_mmr_multiscale_flow\",\"short_decode\":false,\"direct_structured_answer\":false}}"
  local log="$LOG_ROOT/nohup_v442_purefront_${label}_m50_20260712.log"
  echo "START v442 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
  GPUS="$gpu" SAMPLES=50 TASKS="$TASKS" LABEL="v442_purefront_${label}" \
    STAMP="20260712_purefront" POLICY="$policy" bash "$RUNNER" >> "$log" 2>&1
  echo "DONE v442 action=${label} gpu=${gpu} $(date -Is)" | tee -a "$log"
}

worker() {
  local worker_id="$1"
  local gpu="$2"
  local index
  for index in "${!SELECTED[@]}"; do
    if (( index % 2 == worker_id )); then
      run_action "$gpu" "${SELECTED[$index]}"
    fi
  done
}

worker 0 6 &
pid0=$!
worker 1 7 &
pid1=$!
wait "$pid0"
wait "$pid1"
echo "V442 selected Pure M50 frontier complete $(date -Is)"
