#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
RULER_PIDS=(2325362 2325363)

cd "$ROOT"
while true; do
  active=0
  for pid in "${RULER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active=1
      break
    fi
  done
  [[ "$active" -eq 0 ]] && break
  sleep 60
done

exec bash scripts/launch_128k_hot_cache_sweep_20260716.sh

