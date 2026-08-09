#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/parallel_block_retrieval
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=Qwen/Qwen3-0.6B
LONGBENCH=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
BASE_CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record64"
CORPUS="$PROJECT/data/real_longbench_docqa_10m_clean_record480"
BASE_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"
QUERY_PROFILE="$PROJECT/outputs/real_longbench_docqa_10m_allhead_query480_20260714_v1"
RETRIEVAL="$PROJECT/outputs/real_longbench_docqa_10m_allhead_top16_query480_20260714_v1"
ANALYSIS="$PROJECT/outputs/head_sparse_decomposition_10m_query480_20260714_v1"
LOG_ROOT="$PROJECT/outputs/logs/head_sparse_decomposition_10m_query480_20260714_v1"
GPU_IDS="${GPU_IDS:-0,1,2,4,5,6,7}"
WORLD_SIZE="${WORLD_SIZE:-7}"

mkdir -p "$CORPUS" "$QUERY_PROFILE" "$RETRIEVAL" "$ANALYSIS" "$LOG_ROOT"
cd "$PROJECT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "$CORPUS/summary.json" ]]; then
  "$PYTHON" src/prepare_real_longbench_corpus.py \
    --model_name_or_path "$MODEL" \
    --longbench_dir "$LONGBENCH" \
    --output_dir "$CORPUS" \
    --seq_tokens 10000000 \
    --block_tokens 256 \
    --num_queries 480 \
    --max_blocks_per_record 64 \
    --seed 20260710 \
    >"$LOG_ROOT/prepare.log" 2>&1
fi

base_hash=$(sha256sum "$BASE_CORPUS/blocks.npy" | awk '{print $1}')
expanded_hash=$(sha256sum "$CORPUS/blocks.npy" | awk '{print $1}')
echo "base_blocks_sha256=$base_hash" | tee "$LOG_ROOT/block_identity.log"
echo "expanded_blocks_sha256=$expanded_hash" | tee -a "$LOG_ROOT/block_identity.log"
if [[ "$base_hash" != "$expanded_hash" ]]; then
  echo "expanded corpus blocks do not match the frozen K index" >&2
  exit 1
fi

if [[ ! -f "$QUERY_PROFILE/query_profiles.pt" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$WORLD_SIZE" \
    src/profile_all_head_queries_reuse.py \
      --model_name_or_path "$MODEL" \
      --corpus_dir "$CORPUS" \
      --base_profile_dir "$BASE_PROFILE" \
      --output_dir "$QUERY_PROFILE" \
      --query_vector_tokens 16 \
      --dtype float16 \
      --attn_implementation sdpa \
      >"$LOG_ROOT/query_profile.log" 2>&1
fi

if [[ ! -f "$RETRIEVAL/per_head_topk.npz" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$WORLD_SIZE" \
    src/run_all_head_consensus_retrieval.py \
      --corpus_dir "$CORPUS" \
      --profile_dir "$BASE_PROFILE" \
      --query_profiles "$QUERY_PROFILE/query_profiles.pt" \
      --output_dir "$RETRIEVAL" \
      --target_blocks 39 \
      --top_per_head 16 \
      --query_batch 8 \
      --block_chunk 64 \
      --exclude_block_prefix_tokens 16 \
    >"$LOG_ROOT/retrieval.log" 2>&1
fi

"$PYTHON" src/analyze_all_head_consensus.py \
  --corpus_dir "$CORPUS" \
  --retrieval_dir "$RETRIEVAL" \
  --target_blocks 39 \
  >"$LOG_ROOT/legacy_consensus_analysis.log" 2>&1

"$PYTHON" src/analyze_head_sparse_decomposition.py \
  --topk_npz "$RETRIEVAL/per_head_topk.npz" \
  --queries_jsonl "$CORPUS/queries.jsonl" \
  --output_dir "$ANALYSIS" \
  --num_blocks 39062 \
  --depths 1,2,4,8,16 \
  --cv_splits 500 \
  --random_subsets_per_split 20 \
  --overlap_samples 500000 \
  --seed 20260714 \
  >"$LOG_ROOT/property_analysis.log" 2>&1

echo "$(date -Is) complete analysis=$ANALYSIS"
