#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-7,6,4,2,0,1,3,5}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260712_after_pareto075_v398}"
LABEL="${LABEL:-v398_cost_router075_after_pareto}"
POLICY="${POLICY:-configs/riskkv_task_policy_v398_cost_router075_after_pareto_20260712.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
ALL_TASKS="${ALL_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
mkdir -p "$LOG_DIR"

wait_result_or_skip() {
  local name="$1"
  local result="$2"
  local gate="$3"
  while true; do
    if [[ -f "$result" ]]; then
      echo "READY ${name}: ${result} $(date -Is)"
      return 0
    fi
    if [[ -f "$gate" && "$(cat "$gate")" == "0" ]]; then
      echo "SKIP_SOURCE ${name}: gate failed $(date -Is)"
      return 0
    fi
    echo "WAIT ${name}: ${result} $(date -Is)"
    sleep 300
  done
}

wait_result_or_skip v393 outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn/task_results.csv outputs/riskkv_v19_v393_m100_task_knapsack_v385_m100_gate_20260712.flag
wait_result_or_skip v394 outputs/riskkv_v19_v394_m100_task_knapsack10_exact_20260712_m100_task_knapsack10_exact_v394_m100_bDyn_pDyn/task_results.csv outputs/riskkv_v19_v394_m100_task_knapsack10_exact_m100_gate_20260712.flag
wait_result_or_skip v395 outputs/riskkv_v19_v395_m100_task_knapsack075_exact_20260712_m100_task_knapsack075_exact_v395_m100_bDyn_pDyn/task_results.csv outputs/riskkv_v19_v395_m100_task_knapsack075_exact_m100_gate_20260712.flag
wait_result_or_skip v396 outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn/task_results.csv outputs/riskkv_v19_v396_m100_task_knapsack05_exact_m100_gate_20260712.flag

echo "TRAIN v398 cost-aware router kv<=7.5% $(date -Is)"
"$PY" scripts/train_cost_aware_router_v397_after_pareto_20260712.py \
  --output-dir outputs/riskkv_v19_cost_router_v398_after_pareto075_20260712 \
  --config-out "$POLICY" \
  --kv-limit 0.075 \
  --cal-min-gain -0.001 \
  --test-min-gain 0.0

launch_sync() {
  local samples="$1"
  local out="outputs/riskkv_v19_${LABEL}_${STAMP}_m${samples}_bDyn_pDyn"
  local log="$LOG_DIR/launch_${LABEL}_${STAMP}_m${samples}.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing ${out}/task_results.csv $(date -Is)"
    return 0
  fi
  echo "RUN label=${LABEL} samples=${samples} policy=${POLICY} $(date -Is)"
  env GPUS="$GPUS" LOCK_ROOT="$LOCK_ROOT" GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2500}" GPU_MAX_UTIL="${GPU_MAX_UTIL:-101}" SAMPLES="$samples" LABEL="$LABEL" STAMP="$STAMP" POLICY="$POLICY" TASKS="$ALL_TASKS" bash "$RUNNER" > "$log" 2>&1
}

launch_sync 20

"$PY" - <<'PY'
import csv
from pathlib import Path

full_score = 0.3658
full_online = 3.0988
path = Path("outputs/riskkv_v19_v398_cost_router075_after_pareto_20260712_after_pareto075_v398_m20_bDyn_pDyn/task_results.csv")
flag = Path("outputs/riskkv_v19_v398_cost_router075_after_pareto_m100_gate_20260712.flag")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
score = sum(float(row.get("score") or 0.0) for row in rows) / len(rows)
kv = sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows)
online = sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows)
speed = full_online / max(1e-9, online)
passed = score / full_score >= 0.95 and 0.01 <= kv <= 0.085 and speed >= 2.5
flag.write_text("1\n" if passed else "0\n", encoding="utf-8")
print(f"V398_M20_GATE score={score:.4f} vs_full={score/full_score:.2%} kv={kv:.2%} speed_full={speed:.2f}x passed={passed}")
PY

if [[ "$(cat outputs/riskkv_v19_v398_cost_router075_after_pareto_m100_gate_20260712.flag)" == "1" ]]; then
  launch_sync 100
else
  echo "SKIP v398 M100 because M20 gate failed $(date -Is)"
fi

"$PY" scripts/summarize_lowkv_exploration_20260712.py || true
"$PY" scripts/parse_lowkv_running_progress_20260712.py || true
