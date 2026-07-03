#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

STAMP="${STAMP:-20260703_official_compare}"
SAMPLES="${SAMPLES:-1}"
BUDGETS="${BUDGETS:-256 512 1024 2048}"
OFFICIAL_METHODS="${OFFICIAL_METHODS:-FullKV StreamingLLM SnapKV PyramidKV}"
RUN_OFFICIAL="${RUN_OFFICIAL:-1}"
RUN_OURS="${RUN_OURS:-1}"

mkdir -p outputs/logs outputs/pids

LOG="outputs/logs/official_comparison_detached_${STAMP}.log"
PID_FILE="outputs/pids/official_comparison_detached_${STAMP}.pid"

nohup bash -lc "
set -euo pipefail
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
if [[ \"$RUN_OFFICIAL\" == \"1\" ]]; then
  env SAMPLES=\"$SAMPLES\" BUDGETS=\"$BUDGETS\" METHODS=\"$OFFICIAL_METHODS\" STAMP=\"${STAMP}_official\" \
    bash scripts/run_kvcache_factory_official_longbench_sweep_server.sh
fi
if [[ \"$RUN_OURS\" == \"1\" ]]; then
  env SAMPLES=\"$SAMPLES\" BUDGETS=\"$BUDGETS\" STAMP=\"${STAMP}_ours_llama\" \
    bash scripts/run_ours_adapter_longbench_llama_sweep_server.sh
fi
" > "$LOG" 2>&1 &

echo "$!" > "$PID_FILE"
echo "started pid=$(cat "$PID_FILE") log=$LOG"
