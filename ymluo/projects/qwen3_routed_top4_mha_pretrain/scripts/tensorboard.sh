#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

tensorboard --logdir "${OUTPUT_ROOT}" --host "${HOST}" --port "${PORT}"
