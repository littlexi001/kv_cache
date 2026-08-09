#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1/query_profiles.pt"
LODO="$PROJECT/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1"
OUTPUT="$PROJECT/outputs/selected_head_lodo_scan_10m_query480_20260714_v1"
LOG="$PROJECT/outputs/logs/selected_head_lodo_scan_10m_query480_20260714_v1.log"
GPU_IDS="${GPU_IDS:-7}"
WORLD_SIZE="${WORLD_SIZE:-1}"

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
cd "$PROJECT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$WORLD_SIZE" \
  src/benchmark_selected_head_debiased_retrieval.py \
    --corpus_dir "$CORPUS" \
    --profile_dir "$PROFILE" \
    --query_profiles "$QUERY_PROFILE" \
    --selection_csv "$LODO/unsupervised_head_gate/selected_heads.csv" \
    --full_reference_npz "$LODO/zscore/per_head_topk.npz" \
    --output_dir "$OUTPUT" \
    --gate_feature raw_top1_block_diversity \
    --heads_per_fold 16 \
    --top_per_head 16 \
    --target_blocks 39 \
    --query_batch 8 \
    --block_chunk 64 \
    --exclude_block_prefix_tokens 16 \
    --std_epsilon 1e-4 \
  >"$LOG" 2>&1

echo "$(date -Is) complete output=$OUTPUT"
