#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/workspace/routed_top4_qwen3_0p6b_runs}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

tensorboard --logdir "${OUTPUT_ROOT}" --host "${HOST}" --port "${PORT}"
