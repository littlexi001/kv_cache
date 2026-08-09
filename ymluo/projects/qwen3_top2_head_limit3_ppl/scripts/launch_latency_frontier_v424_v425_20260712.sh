#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
cd "$ROOT"

mkdir -p outputs/logs

"$PY" scripts/train_latency_aware_frontier_router_v424_20260712.py \
  --output-dir outputs/riskkv_v19_v424_latency_frontier_router060_20260712 \
  --config-out configs/riskkv_task_policy_v424_latency_frontier_router060_20260712.json \
  --kv-limit 0.055 \
  --quality-ratio 0.95 \
  --speed-min 6.0 \
  --base-action frontier_050 \
  --n-estimators 220

"$PY" scripts/train_latency_aware_frontier_router_v424_20260712.py \
  --output-dir outputs/riskkv_v19_v425_latency_frontier_router068_20260712 \
  --config-out configs/riskkv_task_policy_v425_latency_frontier_router068_20260712.json \
  --kv-limit 0.055 \
  --quality-ratio 0.95 \
  --speed-min 6.8 \
  --base-action frontier_050 \
  --n-estimators 220
