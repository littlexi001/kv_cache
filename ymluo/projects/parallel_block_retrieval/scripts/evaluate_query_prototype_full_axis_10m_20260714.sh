#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROFILE="$PROJECT/outputs/selected_kv_profile_lodo_10m_20260714_v1"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
TAG="${TAG:-query_prototype_full_axis_10m_20260714_v3}"
OUTPUT="$PROJECT/outputs/$TAG"
LOG="$PROJECT/outputs/logs/$TAG.log"
GPU_ID="${GPU_ID:-1}"
SUPPORT_INDEX_DIR="${SUPPORT_INDEX_DIR:-$PROJECT/outputs/query_prototype_full_axis_10m_20260714_v1/support_indices}"
EXACT_PRIOR_DIR="${EXACT_PRIOR_DIR:-$PROJECT/outputs/selected_head_exact_prior_10m_20260714_v1}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
  src/evaluate_query_prototype_full_axis.py \
    --packed_profile_dir "$PROFILE" \
    --query_profiles "$QUERY_PROFILE" \
    --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
    --queries_jsonl "$CORPUS/queries.jsonl" \
    --full_raw_reference_npz "$LODO/raw/per_head_topk.npz" \
    --full_zscore_reference_npz "$LODO/zscore/per_head_topk.npz" \
    --output_dir "$OUTPUT" \
    --support_index_dir "$SUPPORT_INDEX_DIR" \
    --exact_prior_dir "$EXACT_PRIOR_DIR" \
    --gate_feature raw_top1_block_diversity \
    --heads_per_fold 16 \
    --prototypes 128 \
    --candidate_budgets 128,256,512,1024,2048,4096,8192,9766 \
    --exclude_block_prefix_tokens 16 \
    --block_batch 256 \
    --seed 20260714 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
