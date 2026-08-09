#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

LOG="outputs/logs/watch_train_budget_planner_v11_v12_m100_20260711.log"
mkdir -p outputs/logs configs
exec > >(tee -a "$LOG") 2>&1

echo "START watch_train_budget_planner_v11_v12_m100 $(date -Is)"

budgets=(256 384 512 768 1024 1536 2048 3072)
while true; do
  missing=()
  for b in "${budgets[@]}"; do
    out="outputs/riskkv_v19_budget_sweep_b${b}_20260711_budget_sweep_m100_m100_bDyn_pDyn"
    if [[ ! -s "$out/task_results.csv" ]]; then
      missing+=("b${b}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    echo "ALL_SWEEPS_READY $(date -Is)"
    break
  fi
  echo "WAIT_SWEEPS missing=${missing[*]} $(date -Is)"
  for b in "${budgets[@]}"; do
    out="outputs/riskkv_v19_budget_sweep_b${b}_20260711_budget_sweep_m100_m100_bDyn_pDyn"
    if [[ -f "$out/task_results.csv" ]]; then
      echo "  b${b} rows=$(($(wc -l < "$out/task_results.csv") - 1))"
    else
      log="outputs/logs/riskkv_v19_budget_sweep_b${b}_20260711_budget_sweep_m100_m100_bDyn_pDyn.log"
      echo "  b${b} tail=$(tail -n 1 "$log" 2>/dev/null || true)"
    fi
  done
  sleep 300
done

echo "TRAIN_V11_START $(date -Is)"
python scripts/train_clean_budget_router_v11_m100_20260711.py
python scripts/summarize_clean_budget_router_v11_m100_20260711.py
echo "TRAIN_V11_DONE $(date -Is)"

echo "TRAIN_V12_START $(date -Is)"
python scripts/train_calibrated_budget_planner_v12_m100_20260711.py
echo "TRAIN_V12_DONE $(date -Is)"

python - <<'PY'
import csv
import json
from pathlib import Path

ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
tasks = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

summary_path = ROOT / "outputs/riskkv_v19_budget_planner_v12_m100_compare_summary_20260711.csv"
rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
augmented = []
for row in rows:
    planner_summary = ROOT / row["output_dir"] / "planner_summary.csv"
    if not planner_summary.exists():
        continue
    with planner_summary.open(newline="", encoding="utf-8") as handle:
        details = list(csv.DictReader(handle))
    cal = next(item for item in details if item["split"] == "calibration" and item["task"] == "ALL")
    all_row = next(item for item in details if item["split"] == "all" and item["task"] == "ALL")
    item = dict(row)
    item["calibration_score_ratio_measured"] = cal["learned_vs_reference"]
    item["calibration_kv_relative_measured"] = cal["kv_relative"]
    item["all_score_ratio_measured"] = all_row["learned_vs_reference"]
    item["all_kv_relative_measured"] = all_row["kv_relative"]
    augmented.append(item)

feasible = [row for row in augmented if float(row["calibration_score_ratio_measured"]) >= 1.0]
if not feasible:
    feasible = [row for row in augmented if float(row["calibration_score_ratio_measured"]) >= 0.9975]
if not feasible:
    feasible = augmented
best = min(
    feasible,
    key=lambda row: (
        float(row["calibration_kv_relative_measured"]),
        -float(row["calibration_score_ratio_measured"]),
        -float(row["test_score_ratio"]),
    ),
)

selected_path = ROOT / "outputs/riskkv_v19_budget_planner_v12_m100_selected_20260711.json"
selected_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")

policy = {
    "__extends": "riskkv_task_policy_v300_action_router_extra50_robust_20260711.json",
    "tasks": {
        "*": {
            "ours_learned_router_model_path": f"{best['output_dir']}/model.pkl",
            "ours_learned_router_action_policy_json": f"{best['output_dir']}/action_policy.json",
            "ours_learned_router_confidence_threshold": -1,
            "ours_learned_router_default_action": "reference",
            "ours_learned_router_base_action_router_mode": "v293_rules",
        }
    },
}
for task in tasks:
    policy["tasks"][task] = {
        "action_router": True,
        "action_router_mode": "learned_budget_planner_v2",
    }
policy_path = ROOT / "configs/riskkv_task_policy_v351_budget_planner_v12_m100_calibrated_20260711.json"
policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"selected": best, "policy": str(policy_path.relative_to(ROOT))}, ensure_ascii=False, indent=2))
PY

echo "RUN_V351_M20_START $(date -Is)"
env SAMPLES=20 LABEL=budget_planner_v12_v351 STAMP=20260711_m20 \
  POLICY=configs/riskkv_task_policy_v351_budget_planner_v12_m100_calibrated_20260711.json \
  GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/run_riskkv_task_policy_v19_one_20260709.sh
echo "RUN_V351_M20_DONE $(date -Is)"

python - <<'PY'
import csv
from pathlib import Path
ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
summary = ROOT / "outputs/riskkv_v19_budget_planner_v12_v351_20260711_m20_m20_bDyn_pDyn/summary.csv"
should_run = False
if summary.exists():
    rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
    row = next((item for item in rows if item.get("method") == "ours_page_gather"), rows[-1] if rows else None)
    if row:
        score = float(row.get("score", 0.0) or 0.0)
        keep = float(row.get("keep_fraction", 1.0) or 1.0)
        should_run = score >= 0.40 and keep <= 0.35
(ROOT / "outputs/riskkv_v19_budget_planner_v12_v351_run_m100.flag").write_text("1\n" if should_run else "0\n", encoding="utf-8")
print(f"V351_M100_DECISION should_run={should_run}")
PY

if [[ "$(cat outputs/riskkv_v19_budget_planner_v12_v351_run_m100.flag)" == "1" ]]; then
  echo "RUN_V351_M100_START $(date -Is)"
  env SAMPLES=100 LABEL=budget_planner_v12_v351 STAMP=20260711_m100 \
    POLICY=configs/riskkv_task_policy_v351_budget_planner_v12_m100_calibrated_20260711.json \
    GPUS=0,1,2,3,4,5,6,7 \
    bash scripts/run_riskkv_task_policy_v19_one_20260709.sh
  echo "RUN_V351_M100_DONE $(date -Is)"
else
  echo "SKIP_V351_M100 due to M20 sanity gate $(date -Is)"
fi

echo "DONE watch_train_budget_planner_v11_v12_m100 $(date -Is)"
