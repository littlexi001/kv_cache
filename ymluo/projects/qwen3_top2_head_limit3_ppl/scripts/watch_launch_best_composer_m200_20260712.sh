#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
INTERVAL="${INTERVAL:-300}"
FULL_SCORE="${FULL_SCORE:-0.36581658127460975}"
FULL_ONLINE="${FULL_ONLINE:-3.0988}"
MIN_VS_FULL="${MIN_VS_FULL:-0.95}"
MAX_KV="${MAX_KV:-0.10}"
MIN_SPEED="${MIN_SPEED:-2.5}"
GPUS="${GPUS:-4,5,1,3,6,7,0,2}"
cd "$ROOT"
mkdir -p outputs/logs

select_best() {
  "$PY" - <<'PY'
import csv
import os
from pathlib import Path

full_score = float(os.environ.get("FULL_SCORE", "0.36581658127460975"))
full_online = float(os.environ.get("FULL_ONLINE", "3.0988"))
min_vs_full = float(os.environ.get("MIN_VS_FULL", "0.95"))
max_kv = float(os.environ.get("MAX_KV", "0.10"))
min_speed = float(os.environ.get("MIN_SPEED", "2.5"))

candidates = {
    "v430": {
        "label": "v430_composer_kv06_speed6_task20",
        "stamp": "20260712_v430_m200_auto",
        "policy": "configs/riskkv_task_policy_v430_composer_kv06_speed6_task20_20260712.json",
        "m100": "outputs/riskkv_v19_v430_composer_kv06_speed6_task20_20260712_v430_m100_m100_bDyn_pDyn/task_results.csv",
    },
    "v431": {
        "label": "v431_composer_kv08_speed5_task25",
        "stamp": "20260712_v431_m200_auto",
        "policy": "configs/riskkv_task_policy_v431_composer_kv08_speed5_task25_20260712.json",
        "m100": "outputs/riskkv_v19_v431_composer_kv08_speed5_task25_20260712_v431_m100_m100_bDyn_pDyn/task_results.csv",
    },
    "v435": {
        "label": "v435_dpcomposer_kv10_speed35_task35",
        "stamp": "20260712_v435_m200_auto",
        "policy": "configs/riskkv_task_policy_v435_dpcomposer_kv10_speed35_task35_20260712.json",
        "m100": "outputs/riskkv_v19_v435_dpcomposer_kv10_speed35_task35_20260712_v435_m100_m100_bDyn_pDyn/task_results.csv",
    },
}

eligible = []
for name, item in candidates.items():
    path = Path(item["m100"])
    if not path.exists():
        continue
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        continue
    score = sum(float(row.get("score") or 0.0) for row in rows) / len(rows)
    kv = sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows)
    online = sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows)
    speed = full_online / max(online, 1e-9)
    passed = score >= full_score * min_vs_full and kv <= max_kv and speed >= min_speed
    print(
        f"CAND {name} n={len(rows)} score={score:.4f} vs={score/full_score:.2%} "
        f"kv={kv:.2%} speed={speed:.2f}x passed={passed}",
        file=open("outputs/logs/watch_launch_best_composer_m200_20260712.select.log", "a", encoding="utf-8"),
    )
    if passed:
        eligible.append((score, name, item))

if not eligible:
    raise SystemExit(1)

eligible.sort(reverse=True, key=lambda value: value[0])
_, name, item = eligible[0]
print(f"{name}\t{item['label']}\t{item['stamp']}\t{item['policy']}")
PY
}

while true; do
  echo "CHECK best composer m200 $(date -Is)" >> outputs/logs/watch_launch_best_composer_m200_20260712.log
  if selected="$(select_best 2>/dev/null)"; then
    IFS=$'\t' read -r name label stamp policy <<< "$selected"
    out="outputs/riskkv_v19_${label}_${stamp}_m200_bDyn_pDyn"
    if [[ -f "$out/task_results.csv" ]]; then
      echo "M200 already complete for $name: $out" >> outputs/logs/watch_launch_best_composer_m200_20260712.log
      exit 0
    fi
    if pgrep -af "${label}_${stamp}" >/dev/null; then
      echo "M200 already running for $name: ${label}_${stamp}" >> outputs/logs/watch_launch_best_composer_m200_20260712.log
      sleep "$INTERVAL"
      continue
    fi
    echo "LAUNCH best composer M200 name=$name label=$label policy=$policy $(date -Is)" >> outputs/logs/watch_launch_best_composer_m200_20260712.log
    nohup env \
      GPUS="$GPUS" \
      GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2500}" \
      GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}" \
      SAMPLES=200 \
      LABEL="$label" \
      STAMP="$stamp" \
      POLICY="$policy" \
      TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p" \
      bash scripts/run_riskkv_task_policy_v19_one_20260709.sh \
      > "outputs/logs/nohup_${label}_${stamp}_m200.log" 2>&1 &
    exit 0
  fi
  sleep "$INTERVAL"
done
