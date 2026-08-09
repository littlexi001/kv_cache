#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"

cd "$ROOT"
mkdir -p outputs/logs outputs/riskkv_v19_live_dashboard_20260711

while true; do
  "$PYTHON" scripts/update_riskkv_live_dashboard_20260711.py || true
  sleep "$INTERVAL_SECONDS"
done
