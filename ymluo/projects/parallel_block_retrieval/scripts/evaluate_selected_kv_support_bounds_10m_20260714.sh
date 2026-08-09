#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
BOUNDS="$PROJECT/outputs/selected_kv_support_bounds_10m_20260714_v1"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
OUTPUT="$PROJECT/outputs/selected_kv_support_bound_eval_10m_query480_20260714_v1"
LOG="$PROJECT/outputs/logs/selected_kv_support_bound_eval_10m_query480_20260714_v1.log"
GPU_IDS="${GPU_IDS:-6,7}"
WORLD_SIZE="${WORLD_SIZE:-2}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$WORLD_SIZE" \
  src/evaluate_selected_kv_support_bounds.py \
    --bound_dir "$BOUNDS" \
    --query_profiles "$QUERY_PROFILE" \
    --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
    --queries_jsonl "$CORPUS/queries.jsonl" \
    --full_raw_reference_npz "$LODO/raw/per_head_topk.npz" \
    --output_dir "$OUTPUT" \
    --gate_feature raw_top1_block_diversity \
    --heads_per_fold 16 \
    --query_batch 4 \
    --safety_tolerance 1e-4 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
