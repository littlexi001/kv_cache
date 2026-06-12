#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/artifacts/output_svd_qwen3_0p6b}"
mkdir -p "$OUTPUT_DIR"
nohup python3 -u fdong_seq_compress/src/collect_output_svd_basis.py \
  --model-path "${MODEL_PATH:-fdong/Qwen3-0.6B}" \
  --device "${DEVICE:-mps}" \
  --dtype "${DTYPE:-float16}" \
  --text-globs "${TEXT_GLOBS:-fdong_seq_compress/data/synthetic_texts/*.txt}" \
  --chunk-tokens "${CHUNK_TOKENS:-1024}" \
  --chunk-stride "${CHUNK_STRIDE:-768}" \
  --max-chunks "${MAX_CHUNKS:-64}" \
  --samples-per-chunk "${SAMPLES_PER_CHUNK:-64}" \
  --output-dir "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/build.log" 2>&1 &
PID=$!
echo "$PID" > "$OUTPUT_DIR/build.pid"
echo "Started PID $PID"
echo "Log: $OUTPUT_DIR/build.log"
echo "Follow: tail -f $OUTPUT_DIR/build.log"
