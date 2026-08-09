#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
SOURCE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
OUTPUT="$PROJECT/outputs/selected_kv_profile_lodo_10m_20260714_v1"
LOG="$PROJECT/outputs/logs/selected_kv_profile_lodo_10m_20260714_v1.log"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

"$PYTHON" src/pack_selected_kv_profile.py \
  --source_profile_dir "$SOURCE" \
  --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
  --output_dir "$OUTPUT" \
  --gate_feature raw_top1_block_diversity \
  --heads_per_fold 16 \
  --block_chunk 128 \
  --workers 2 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
