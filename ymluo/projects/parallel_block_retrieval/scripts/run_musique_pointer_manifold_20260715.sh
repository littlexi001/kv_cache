#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

CORPUS_DIR="data/musique_official_10m_aligned_2000_v3"
K_PROFILE="outputs/musique_official_test500_selectedlayer_k_svd32_10m_20260715_v1"
Q_PROFILE="outputs/musique_official_all2000_allhead_stepq_10m_20260715_v1"
OUTPUT_DIR="outputs/musique_official_pointer_query_manifold_20260715_v1"

# Frozen LongBench dataset-LODO heads; this fold excludes all LongBench MuSiQue queries.
SELECTED_HEADS="3:10,21:8,13:6,16:15,14:15,11:3,20:15,8:4,20:6,26:7,16:14,14:14,6:7,14:6,16:13,25:9"

if [[ ! -f "${Q_PROFILE}/step_query_profiles.pt" ]]; then
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    src/profile_step_state_q.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --base_profile_dir "${K_PROFILE}" \
    --step_queries_path "${CORPUS_DIR}/step_queries.jsonl" \
    --output_dir "${Q_PROFILE}" \
    --splits train,dev,test \
    --task_types multihop \
    --query_vector_tokens 16 \
    --dtype float16 \
    --attn_implementation sdpa
fi

"${PYTHON}" src/analyze_state_pointer_query_manifold.py \
  --step_profile "${Q_PROFILE}/step_query_profiles.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --selected_heads "${SELECTED_HEADS}" \
  --train_splits train,dev \
  --test_splits test \
  --prototypes 128 \
  --fit_sample 4096 \
  --kmeans_iterations 12 \
  --device cuda
