#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
INTERVAL=${INTERVAL:-180}
LOG=${LOG:-logs/watch_and_launch_m100_if_m50_stable_20260709.log}
MAX_WAIT_LOOPS=${MAX_WAIT_LOOPS:-720}
MIN_MARGIN_VS_FULL=${MIN_MARGIN_VS_FULL:--0.003}
MIN_MARGIN_VS_V37=${MIN_MARGIN_VS_V37:--0.001}

cd "$ROOT"
mkdir -p logs

m50_ready() {
  [[ -f outputs/riskkv_fullkv_m50_same_samples_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v37_high_quality_m50_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v52_consistency_quality_m50_20260709/summary.csv ]] &&
    [[ -f outputs/riskkv_v53_consistency_quality_qasper_full_m50_20260709/summary.csv ]]
}

m100_done_or_running() {
  [[ -f outputs/riskkv_v53_consistency_quality_qasper_full_m100_20260709/summary.csv ]] ||
    pgrep -af "riskkv_.*m100_20260709" >/dev/null 2>&1
}

free_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2 + 0) < 1000 && ($3 + 0) < 20) print $1}'
}

stable_enough() {
  "$PY" - <<'PY'
import csv
import os
from pathlib import Path

root = Path(".")
paths = {
    "full": root / "outputs/riskkv_fullkv_m50_same_samples_20260709/summary.csv",
    "v37": root / "outputs/riskkv_v37_high_quality_m50_20260709/summary.csv",
    "v52": root / "outputs/riskkv_v52_consistency_quality_m50_20260709/summary.csv",
    "v53": root / "outputs/riskkv_v53_consistency_quality_qasper_full_m50_20260709/summary.csv",
}

def score(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("benchmark") == "ALL" and row.get("task") == "ALL":
                return float(row["score"])
            if row.get("benchmark") == "longbench" and row.get("task") == "ALL":
                fallback = float(row["score"])
        return fallback

scores = {name: score(path) for name, path in paths.items()}
margin_full = float(os.environ.get("MIN_MARGIN_VS_FULL", "-0.003"))
margin_v37 = float(os.environ.get("MIN_MARGIN_VS_V37", "-0.001"))
best_consistency = max(scores["v52"], scores["v53"])
stable = (best_consistency >= scores["full"] + margin_full) or (best_consistency >= scores["v37"] + margin_v37)
print(
    "m50_scores "
    + " ".join(f"{key}={value:.6f}" for key, value in scores.items())
    + f" best_consistency={best_consistency:.6f} stable={int(stable)}"
)
raise SystemExit(0 if stable else 1)
PY
}

{
  echo "[$(date)] m100 stable watcher started"
  for ((i = 1; i <= MAX_WAIT_LOOPS; i++)); do
    if m100_done_or_running; then
      echo "[$(date)] m100 already done or running"
      exit 0
    fi
    if ! m50_ready; then
      echo "[$(date)] loop=$i m50 not ready"
      sleep "$INTERVAL"
      continue
    fi
    if ! stable_enough; then
      echo "[$(date)] m50 consistency policies not stable enough; not launching m100"
      exit 0
    fi
    mapfile -t gpus < <(free_gpus)
    echo "[$(date)] loop=$i m50 stable free_gpus=${gpus[*]:-none}"
    if (( ${#gpus[@]} >= 4 )); then
      GPU_FULL=${gpus[0]} GPU_V37=${gpus[1]} GPU_V52=${gpus[2]} GPU_V53=${gpus[3]} \
        bash scripts/run_riskkv_v37_v52_v53_m100_20260709.sh
      echo "[$(date)] launched m100 on GPUs ${gpus[0]} ${gpus[1]} ${gpus[2]} ${gpus[3]}"
      exit 0
    fi
    sleep "$INTERVAL"
  done
  echo "[$(date)] m100 stable watcher timed out without launching"
} >> "$LOG" 2>&1
