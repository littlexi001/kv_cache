#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROFILE="$PROJECT/outputs/selected_kv_profile_lodo_10m_20260714_v1"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
TAG="${TAG:-selected_head_exact_prior_10m_20260714_v2}"
OUTPUT="$PROJECT/outputs/$TAG"
LOG="$PROJECT/outputs/logs/$TAG.log"
GPU_ID="${GPU_ID:-1}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
  src/build_selected_head_exact_prior_profile.py \
    --packed_profile_dir "$PROFILE" \
    --query_profiles "$QUERY_PROFILE" \
    --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
    --reference_npz "$LODO/zscore/per_head_topk.npz" \
    --output_dir "$OUTPUT" \
    --gate_feature raw_top1_block_diversity \
    --heads_per_fold 16 \
    --query_batch 8 \
    --block_chunk 64 \
    --exclude_block_prefix_tokens 16 \
    --std_epsilon 1e-4 \
    --save_raw_scores \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
