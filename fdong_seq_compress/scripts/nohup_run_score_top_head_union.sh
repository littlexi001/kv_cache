#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/score_top_head_union_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"

nohup python3 -u fdong_seq_compress/src/run_score_top_task_ablation.py \
  --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
  --data-path "${DATA_PATH:-ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl}" \
  --sample-id "${SAMPLE_ID:-niah_len2000_depth25}" \
  --device "${DEVICE:-mps}" \
  --dtype "${DTYPE:-float16}" \
  --score-ratio "${SCORE_RATIO:-0.02}" \
  --position-ratio "${POSITION_RATIO:-0.01}" \
  --excluded-categories "" \
  --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/run.log" 2>&1 &

PID=$!
echo "$PID" > "$OUTPUT_DIR/run.pid"
echo "Started PID $PID"
echo "Log: $OUTPUT_DIR/run.log"
echo "Output: $OUTPUT_DIR"
echo "Follow: tail -f $OUTPUT_DIR/run.log"
