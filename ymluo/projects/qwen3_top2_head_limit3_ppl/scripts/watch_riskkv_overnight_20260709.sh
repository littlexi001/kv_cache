#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
if [[ -f /home/fdong/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/fdong/miniconda3/etc/profile.d/conda.sh
  conda activate moe || true
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi
LOG="outputs/logs/riskkv_overnight_watch_20260709.log"
SUMMARY="outputs/riskkv_flow_v12_summary_20260709.csv"
mkdir -p outputs/logs

for iter in $(seq 1 72); do
  {
    echo "=== WATCH iter=$iter $(date -Is) ==="
    "$PYTHON_BIN" scripts/summarize_riskkv_flow_v12_20260709.py || true
    if [[ -f "$SUMMARY" ]]; then
      echo "summary_rows=$(($(wc -l < "$SUMMARY") - 1)) path=$SUMMARY"
      tail -20 "$SUMMARY"
    fi
    echo "--- active riskkv jobs ---"
    ps -ef | grep -E 'riskkv_flow|multiscale_flow|run_controlled_public_kv_benchmark' | grep -v grep || true
    echo "--- gpu ---"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true
  } >> "$LOG" 2>&1
  sleep 600
done
