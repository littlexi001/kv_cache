#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/score_top_stability}"
mkdir -p "$OUTPUT_DIR"

nohup python3 -u fdong_seq_compress/src/analyze_score_top_stability.py \
  --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
  --data-path "${DATA_PATH:-ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}" \
  --sample-ids "${SAMPLE_IDS:-default}" \
  --score-ratio "${SCORE_RATIO:-0.02}" \
  --position-ratio "${POSITION_RATIO:-0.01}" \
  --device "${DEVICE:-mps}" \
  --dtype "${DTYPE:-float16}" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/run.log" 2>&1 &

PID=$!
echo "$PID" > "$OUTPUT_DIR/run.pid"
echo "Started PID $PID"
echo "Output: $OUTPUT_DIR"
echo "Follow: tail -f $OUTPUT_DIR/run.log"
