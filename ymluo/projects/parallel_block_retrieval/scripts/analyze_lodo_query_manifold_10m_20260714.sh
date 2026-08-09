#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
OUTPUT="$PROJECT/outputs/lodo_query_manifold_10m_query480_20260714_v1"
LOG="$PROJECT/outputs/logs/lodo_query_manifold_10m_query480_20260714_v1.log"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"

"$PYTHON" src/analyze_lodo_query_manifold.py \
  --query_profiles "$QUERY_PROFILE" \
  --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --fold_reference_npz "$LODO/raw/per_head_topk.npz" \
  --output_dir "$OUTPUT" \
  --gate_feature raw_top1_block_diversity \
  --heads_per_fold 16 \
  --prototype_counts 8,32,128 \
  --subspace_ranks 4,8,16,24 \
  --max_train_tokens 8192 \
  --seed 20260714 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
