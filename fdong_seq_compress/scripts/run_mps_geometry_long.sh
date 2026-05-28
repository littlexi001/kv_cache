#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-fdong_seq_compress/outputs/qwen3_geometry_mps_long_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-mps}"
DTYPE="${DTYPE:-float16}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-512,1024,2048,4096,8192,12000}"

# Full all-layer/all-head runs can take a while because SVD is still computed on CPU.
# Override these from the shell if you want a faster pilot, for example LAYERS=0,7,15,27 HEADS=0,1.
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
KINDS="${KINDS:-K,V}"
BLOCK_SIZES="${BLOCK_SIZES:-4,8,16,32,64,128}"
SUBSPACE_RANK="${SUBSPACE_RANK:-16}"
ENERGY_THRESHOLDS="${ENERGY_THRESHOLDS:-0.90,0.95,0.99}"

mkdir -p "$OUTPUT_DIR"

echo "Writing run log to $OUTPUT_DIR/run.log"
echo "Model: $MODEL_PATH"
echo "Text: $TEXT_PATH"
echo "Output: $OUTPUT_DIR"
echo "Device: $DEVICE dtype=$DTYPE"
echo "Prefixes: $PREFIX_LENGTHS"

{
  echo "=== run_mps_geometry_long start $(date) ==="
  echo "MODEL_PATH=$MODEL_PATH"
  echo "TEXT_PATH=$TEXT_PATH"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "DEVICE=$DEVICE"
  echo "DTYPE=$DTYPE"
  echo "MAX_TOKENS=$MAX_TOKENS"
  echo "PREFIX_LENGTHS=$PREFIX_LENGTHS"
  echo "LAYERS=$LAYERS"
  echo "HEADS=$HEADS"
  echo "KINDS=$KINDS"
  echo "BLOCK_SIZES=$BLOCK_SIZES"
  echo "SUBSPACE_RANK=$SUBSPACE_RANK"
  echo "ENERGY_THRESHOLDS=$ENERGY_THRESHOLDS"
  echo

  python3 -c "import torch; print('mps built', torch.backends.mps.is_built()); print('mps available', torch.backends.mps.is_available()); print('cuda available', torch.cuda.is_available())"
  echo

  time python3 fdong_seq_compress/src/run_prefix_geometry.py \
    --model-path "$MODEL_PATH" \
    --text-path "$TEXT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-tokens "$MAX_TOKENS" \
    --prefix-lengths "$PREFIX_LENGTHS" \
    --layers "$LAYERS" \
    --heads "$HEADS" \
    --kinds "$KINDS" \
    --energy-thresholds "$ENERGY_THRESHOLDS" \
    --block-sizes "$BLOCK_SIZES" \
    --subspace-rank "$SUBSPACE_RANK"

  echo "=== run_mps_geometry_long done $(date) ==="
} 2>&1 | tee "$OUTPUT_DIR/run.log"

echo
echo "Done. Key files:"
echo "  $OUTPUT_DIR/run.log"
echo "  $OUTPUT_DIR/timings.csv"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/metrics_by_prefix_layer_head.csv"
