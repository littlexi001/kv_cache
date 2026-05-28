#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-fdong/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-fdong_seq_compress/data/synthetic_texts/long_english_article_01.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
PREFIX_LENGTHS="${PREFIX_LENGTHS:-128,256,512,1024}"
LAYERS="${LAYERS:-all}"
HEADS="${HEADS:-all}"
KINDS="${KINDS:-K,V}"
ENERGY_THRESHOLDS="${ENERGY_THRESHOLDS:-0.90,0.95,0.99}"
BLOCK_SIZES="${BLOCK_SIZES:-4,8,16,32,64}"
SUBSPACE_RANK="${SUBSPACE_RANK:-16}"

python3 fdong_seq_compress/src/run_prefix_geometry.py \
  --model-path "$MODEL_PATH" \
  --text-path "$TEXT_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-tokens "$MAX_TOKENS" \
  --prefix-lengths "$PREFIX_LENGTHS" \
  --layers "$LAYERS" \
  --heads "$HEADS" \
  --kinds "$KINDS" \
  --energy-thresholds "$ENERGY_THRESHOLDS" \
  --block-sizes "$BLOCK_SIZES" \
  --subspace-rank "$SUBSPACE_RANK" \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}

