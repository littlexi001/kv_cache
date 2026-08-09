#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROFILE="$PROJECT/outputs/selected_kv_profile_lodo_10m_20260714_v1"
SUPPORT="$PROJECT/outputs/query_prototype_full_axis_10m_20260714_v1/support_indices"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
EXACT="$PROJECT/outputs/selected_head_exact_prior_10m_20260714_v2"
TAG="${TAG:-prototype_exact_rerank_lodo_10m_20260714_v1}"
OUTPUT="$PROJECT/outputs/$TAG"
LOG="$PROJECT/outputs/logs/$TAG.log"
GPU_ID="${GPU_ID:-1}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" \
  src/evaluate_prototype_exact_rerank_lodo.py \
    --packed_profile_dir "$PROFILE" \
    --support_index_dir "$SUPPORT" \
    --query_profiles "$QUERY_PROFILE" \
    --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
    --queries_jsonl "$CORPUS/queries.jsonl" \
    --zscore_reference_npz "$LODO/zscore/per_head_topk.npz" \
    --exact_score_profile_dir "$EXACT" \
    --output_dir "$OUTPUT" \
    --gate_feature raw_top1_block_diversity \
    --heads_per_fold 16 \
    --prototypes 128 \
    --candidate_budgets 2048,4096,8192,9766 \
    --top_per_head 16 \
    --target_blocks 39 \
    --std_epsilon 1e-4 \
    --seed 20260714 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
