#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROFILE="$PROJECT/outputs/selected_kv_profile_lodo_10m_20260714_v1"
OUTPUT="$PROJECT/outputs/selected_kv_support_bounds_10m_20260714_v1"
LOG="$PROJECT/outputs/logs/selected_kv_support_bounds_10m_20260714_v1.log"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

"$PYTHON" src/build_selected_kv_support_bounds.py \
  --packed_profile_dir "$PROFILE" \
  --output_dir "$OUTPUT" \
  --segments 1,2,4,8,16 \
  --exclude_block_prefix_tokens 16 \
  --block_chunk 64 \
  --workers 2 \
  --radius_relative_margin 1e-6 \
  --radius_absolute_margin 1e-6 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
