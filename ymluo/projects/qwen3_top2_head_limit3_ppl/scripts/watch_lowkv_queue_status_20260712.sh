#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
INTERVAL="${INTERVAL:-300}"
OUT="${OUT:-outputs/lowkv_queue_status_20260712.txt}"
cd "$ROOT"
mkdir -p "$(dirname "$OUT")" outputs/logs

while true; do
  {
    echo "===== lowkv queue status $(date -Is) ====="
    "$PY" scripts/summarize_lowkv_queue_20260712.py || true
    echo
    echo "===== live partial by run ====="
    "$PY" scripts/summarize_live_log_metrics_20260712.py \
      --runs v427_m200,v428_m200,full_m200,v429_m100,v430_m100,v431_m100,v433_m100,v434_m100,v435_m100,v427_ruler,full_ruler,v436_ruler || true
    echo
    echo "===== matched ordinal partial ====="
    "$PY" scripts/summarize_live_matched_metrics_20260712.py \
      --baseline full_m200 \
      --runs v427_m200,v428_m200 \
      --match ordinal || true
    "$PY" scripts/summarize_live_matched_metrics_20260712.py \
      --baseline full_ruler \
      --runs v427_ruler,v436_ruler \
      --match ordinal || true
    echo
    echo "===== waiter tails ====="
    for log in \
      outputs/logs/nohup_v433_dpcomposer_kv06_speed6_task20_20260712_v433_m100.log \
      outputs/logs/nohup_v434_dpcomposer_kv08_speed5_task25_20260712_v434_m100.log \
      outputs/logs/nohup_v435_dpcomposer_kv10_speed35_task35_20260712_v435_m100.log \
      outputs/logs/nohup_v436_ruler_lowkv_waiter_20260712.log \
      outputs/logs/watch_launch_best_composer_m200_20260712.log \
      outputs/logs/watch_launch_best_composer_m200_20260712.select.log; do
      echo "--- $log"
      tail -n 4 "$log" 2>/dev/null || true
    done
  } > "$OUT.tmp"
  mv "$OUT.tmp" "$OUT"
  sleep "$INTERVAL"
done
