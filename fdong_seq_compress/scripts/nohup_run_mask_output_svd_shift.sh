#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/mask_output_svd_shift_${TIMESTAMP}}"
mkdir -p "$OUTPUT_DIR"
nohup python3 -u fdong_seq_compress/src/analyze_mask_output_svd_shift.py \
  --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
  --artifact-dir "${ARTIFACT_DIR:-fdong_seq_compress/artifacts/output_svd_qwen3_0p6b}" \
  --device "${DEVICE:-mps}" \
  --dtype "${DTYPE:-float16}" \
  --score-ratio "${SCORE_RATIO:-0.02}" \
  --excluded-categories "${EXCLUDED_CATEGORIES:-answer,front,end,other}" \
  --ig-steps "${IG_STEPS:-33}" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/run.log" 2>&1 &
PID=$!
echo "$PID" > "$OUTPUT_DIR/run.pid"
echo "Started PID $PID"
echo "Log: $OUTPUT_DIR/run.log"
echo "Follow: tail -f $OUTPUT_DIR/run.log"
