#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
PY="/home/fdong/miniconda3/envs/moe/bin/python"
THRESH_SCORE="0.353371"
THRESH_KEEP="0.340000"
THRESH_ONLINE="1.450000"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

cd "$ROOT"
mkdir -p logs

declare -a CANDIDATES=(
  "v189_gov128_hotpot2048_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v189_gov128_hotpot2048_direct_heads_20260710.json"
  "v190_gov64_hotpot2048_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v190_gov64_hotpot2048_direct_heads_20260710.json"
  "v191_gov128_hotpot3072_direct_heads|20260710_m50_quality_backup|configs/riskkv_task_policy_v191_gov128_hotpot3072_direct_heads_20260710.json"
  "v192_gov96_hotpot2048_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v192_gov96_hotpot2048_direct_heads_20260710.json"
  "v193_gov128_summary64_hotpot2048_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v193_gov128_summary64_hotpot2048_direct_heads_20260710.json"
  "v194_gov128_hotpot_safe_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v194_gov128_hotpot_safe_direct_heads_20260710.json"
  "v195_gov96_hotpot_safe_direct_heads|20260710_m50_speed_target|configs/riskkv_task_policy_v195_gov96_hotpot_safe_direct_heads_20260710.json"
)

summary_path() {
  local label="$1"
  local stamp="$2"
  echo "outputs/riskkv_v19_${label}_${stamp}_m50_bDyn_pDyn/summary.csv"
}

passes_thresholds() {
  local summary="$1"
  "$PY" - "$summary" "$THRESH_SCORE" "$THRESH_KEEP" "$THRESH_ONLINE" <<'PY'
import csv
import sys

summary, score_t, keep_t, online_t = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
with open(summary, newline="") as f:
    for row in csv.DictReader(f):
        if row["benchmark"] == "ALL" and row["task"] == "ALL":
            score = float(row["score"])
            keep = float(row["mean_keep_fraction"])
            online = float(row["mean_online_seconds"])
            ok = score >= score_t and keep <= keep_t and online <= online_t
            print(f"score={score:.6f} keep={keep:.6f} online={online:.6f} ok={int(ok)}")
            sys.exit(0 if ok else 1)
raise SystemExit(1)
PY
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F, '$2 + 0 < 1000 {gsub(/ /, "", $1); print $1; exit}'
}

launch_m100() {
  local label="$1"
  local policy="$2"
  local gpu="$3"
  local m100_label="${label}_m100_auto"
  local m100_stamp="20260710_m100_auto"
  local out="outputs/riskkv_v19_${m100_label}_${m100_stamp}_m100_bDyn_pDyn"
  if [[ -f "$out/summary.csv" ]] || pgrep -af "$m100_label" >/dev/null; then
    echo "m100 already exists or running for $label"
    return 0
  fi
  echo "Launching m100 for $label on GPU $gpu with $policy"
  setsid -f env GPUS="$gpu" SAMPLES=100 LABEL="$m100_label" STAMP="$m100_stamp" POLICY="$policy" \
    bash scripts/run_riskkv_task_policy_v19_one_20260709.sh \
    > "logs/${m100_label}_${m100_stamp}_launcher.log" 2>&1 < /dev/null
}

echo "watcher started at $(date)"
while true; do
  for item in "${CANDIDATES[@]}"; do
    IFS='|' read -r label stamp policy <<<"$item"
    summary="$(summary_path "$label" "$stamp")"
    if [[ ! -f "$summary" ]]; then
      continue
    fi
    if passes_thresholds "$summary"; then
      gpu="$(free_gpu || true)"
      if [[ -n "${gpu:-}" ]]; then
        launch_m100 "$label" "$policy" "$gpu"
        echo "watcher finished after launching $label m100"
        exit 0
      fi
      echo "candidate $label passed but no free GPU; waiting"
    else
      echo "candidate $label completed but did not pass thresholds"
    fi
  done
  sleep "$SLEEP_SECONDS"
done
