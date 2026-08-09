#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"
MONITOR_LOG="$LOG_ROOT/monitor_speed_caps_20260711.log"

log() {
  echo "[$(date -Is)] $*" | tee -a "$MONITOR_LOG"
}

metric_csv() {
  local tag="$1"
  /home/fdong/miniconda3/envs/moe/bin/python - "$tag" <<'PY'
import csv
import sys
from pathlib import Path

tag = sys.argv[1]
path = Path("outputs") / tag / "summary.csv"
if not path.exists():
    raise SystemExit(2)
for row in csv.DictReader(path.open()):
    if row["benchmark"] == "ALL" and row["task"] == "ALL" and row["method"] == "ours_page_gather":
        print(row["score"], row["mean_online_seconds"], row["mean_keep_fraction"])
        break
else:
    raise SystemExit(3)
PY
}

wait_for_summary() {
  local tag="$1"
  local path="outputs/$tag/summary.csv"
  for _ in $(seq 1 720); do
    if [[ -f "$path" ]]; then
      return 0
    fi
    sleep 60
  done
  return 1
}

maybe_launch_m100() {
  local tag="$1"
  local policy="$2"
  local label="$3"
  local stamp="$4"

  log "waiting tag=$tag"
  if ! wait_for_summary "$tag"; then
    log "timeout tag=$tag"
    return 0
  fi

  read -r score online keep < <(metric_csv "$tag")
  log "summary tag=$tag score=$score online=$online keep=$keep"

  if /home/fdong/miniconda3/envs/moe/bin/python - "$score" "$online" "$keep" <<'PY'
import sys
score = float(sys.argv[1])
online = float(sys.argv[2])
keep = float(sys.argv[3])
ok = score >= 0.355 and online <= 1.20 and keep <= 0.30
raise SystemExit(0 if ok else 1)
PY
  then
    :
  else
    log "skip m100 label=$label because m20 gate failed"
    return 0
  fi

  local out="outputs/riskkv_v19_${label}_${stamp}_m100_bDyn_pDyn"
  if [[ -f "$out/summary.csv" ]]; then
    log "skip m100 label=$label because summary already exists"
    return 0
  fi

  log "launch m100 label=$label policy=$policy"
  setsid env GPUS=1,3,5 SAMPLES=100 POLICY="$policy" LABEL="$label" STAMP="$stamp" \
    bash scripts/run_riskkv_task_policy_v19_one_20260709.sh \
    > "$LOG_ROOT/launcher_${label}.log" 2>&1 < /dev/null &
}

maybe_launch_m100 \
  "riskkv_v19_v224_structured_speed_caps_m20_20260711_speed_caps_m20_bDyn_pDyn" \
  "configs/riskkv_task_policy_v224_structured_speed_caps_20260711.json" \
  "v226_structured_speed_caps_full_m100" \
  "20260711_speed_caps_full"

maybe_launch_m100 \
  "riskkv_v19_v225_structured_speed_caps_aggr_m20_20260711_speed_caps_m20_bDyn_pDyn" \
  "configs/riskkv_task_policy_v225_structured_speed_caps_aggr_20260711.json" \
  "v227_structured_speed_caps_aggr_full_m100" \
  "20260711_speed_caps_full"

log "monitor complete"
