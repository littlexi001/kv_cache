#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-2,4,7}"
RUNNER="${RUNNER:-scripts/run_riskkv_task_policy_v19_one_20260709.sh}"
STAMP="${STAMP:-20260712_after_v385_v389_v392}"
LABEL="${LABEL:-v392_after_v385_v389_winner}"
POLICY="${POLICY:-configs/riskkv_task_policy_v392_task_gated_winner_after_v385_v389_20260712.json}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
ALL_TASKS="${ALL_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
V385_RESULTS="${V385_RESULTS:-outputs/riskkv_v19_v385_quality10_v384_plus_v363_qmsumlow_20260712_quality10_v384_plus_v363_qmsumlow_v385_m100_bDyn_pDyn/task_results.csv}"
V389_RESULTS="${V389_RESULTS:-outputs/riskkv_v19_v389_m100_task_knapsack_v2_20260712_m100_task_knapsack_v2_v389_m100_bDyn_pDyn/task_results.csv}"
mkdir -p "$LOG_DIR"

launch_sync() {
  local samples="$1"
  local out="outputs/riskkv_v19_${LABEL}_${STAMP}_m${samples}_bDyn_pDyn"
  local log="$LOG_DIR/launch_${LABEL}_${STAMP}_m${samples}.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing ${out}/task_results.csv $(date -Is)"
    return 0
  fi
  echo "RUN label=${LABEL} samples=${samples} policy=${POLICY} $(date -Is)"
  env \
    GPUS="$GPUS" \
    LOCK_ROOT="$LOCK_ROOT" \
    SAMPLES="$samples" \
    LABEL="$LABEL" \
    STAMP="$STAMP" \
    POLICY="$POLICY" \
    TASKS="$ALL_TASKS" \
    bash "$RUNNER" > "$log" 2>&1
}

echo "WAIT v385/v389 M100 results $(date -Is)"
until [[ -f "$V385_RESULTS" && -f "$V389_RESULTS" ]]; do
  "$PY" scripts/parse_lowkv_running_progress_20260712.py >/dev/null 2>&1 || true
  sleep 300
done

echo "TRAIN v392 after v385/v389 M100 $(date -Is)"
"$PY" scripts/train_winner_router_v392_after_v385_v389_20260712.py > "$LOG_DIR/train_v392_after_v385_v389_20260712.log" 2>&1

"$PY" - <<'PY'
import csv
import json
from pathlib import Path

summary = Path("outputs/riskkv_v19_winner_router_v392_after_v385_v389_20260712/winner_summary.csv")
meta = Path("outputs/riskkv_v19_winner_router_v392_after_v385_v389_20260712/metadata.json")
flag = Path("outputs/riskkv_v19_v392_after_v385_v389_offline_gate_20260712.flag")
rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
meta_obj = json.loads(meta.read_text(encoding="utf-8"))
selected_tasks = meta_obj.get("selected_tasks", [])
all_row = next(row for row in rows if row["split"] == "all" and row["task"] == "ALL")
cal_row = next(row for row in rows if row["split"] == "calibration" and row["task"] == "ALL")
test_row = next(row for row in rows if row["split"] == "test" and row["task"] == "ALL")
score = float(all_row["score"])
kv = float(all_row["kv"])
speed = float(all_row["speed_vs_full"])
all_gain = float(all_row["gain"])
cal_gain = float(cal_row["gain"])
test_gain = float(test_row["gain"])
passed = bool(selected_tasks) and score >= 0.3906 and all_gain >= 0.0 and cal_gain >= -0.001 and test_gain >= 0.0 and kv <= 0.10 and speed >= 2.5
flag.write_text("1\n" if passed else "0\n", encoding="utf-8")
print(
    f"V392_OFFLINE_GATE score={score:.4f} gain={all_gain:+.4f} "
    f"cal_gain={cal_gain:+.4f} test_gain={test_gain:+.4f} kv={kv:.2%} speed={speed:.2f}x "
    f"tasks={selected_tasks} passed={passed}"
)
PY

if [[ "$(cat outputs/riskkv_v19_v392_after_v385_v389_offline_gate_20260712.flag)" != "1" ]]; then
  echo "SKIP v392 runs because offline gate failed $(date -Is)"
  "$PY" scripts/summarize_lowkv_exploration_20260712.py || true
  exit 0
fi

launch_sync 20

"$PY" - <<'PY'
import csv
from pathlib import Path

full_score = 0.3658
full_online = 3.0988
path = Path("outputs/riskkv_v19_v392_after_v385_v389_winner_20260712_after_v385_v389_v392_m20_bDyn_pDyn/task_results.csv")
flag = Path("outputs/riskkv_v19_v392_after_v385_v389_winner_m100_gate_20260712.flag")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
score = sum(float(row.get("score") or 0.0) for row in rows) / len(rows)
kv = sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows)
online = sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows)
speed = full_online / max(1e-9, online)
passed = score / full_score >= 0.95 and 0.01 <= kv <= 0.105 and speed >= 2.5
flag.write_text("1\n" if passed else "0\n", encoding="utf-8")
print(f"V392_M20_GATE score={score:.4f} vs_full={score/full_score:.2%} kv={kv:.2%} speed_full={speed:.2f}x passed={passed}")
PY

if [[ "$(cat outputs/riskkv_v19_v392_after_v385_v389_winner_m100_gate_20260712.flag)" == "1" ]]; then
  launch_sync 100
else
  echo "SKIP v392 M100 because M20 gate failed $(date -Is)"
fi

"$PY" scripts/summarize_lowkv_exploration_20260712.py || true
"$PY" scripts/parse_lowkv_running_progress_20260712.py || true
