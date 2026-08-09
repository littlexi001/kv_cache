#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
INTERVAL="${INTERVAL:-600}"
ROUNDS="${ROUNDS:-72}"

for _ in $(seq 1 "$ROUNDS"); do
  "$PY" scripts/update_lowkv_overnight_report_20260712.py || true
  sleep "$INTERVAL"
done
