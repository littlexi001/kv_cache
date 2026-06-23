#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${LOGDIR:-${PROJECT_DIR}/output/kv_head_gate_runs}"
PORT="${PORT:-6006}"

tensorboard --logdir "${LOGDIR}" --host 0.0.0.0 --port "${PORT}"
