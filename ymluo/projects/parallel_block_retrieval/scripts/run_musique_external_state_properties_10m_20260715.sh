#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-1,2,3,4,5,6}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

CORPUS_DIR="data/musique_official_10m_aligned_2000_v3"
STEP_PATH="${CORPUS_DIR}/step_queries.jsonl"
K_PROFILE="outputs/musique_official_test500_selectedlayer_k_svd32_10m_20260715_v1"
Q_PROFILE="outputs/musique_official_test500_allhead_stepq_10m_20260715_v1"
TRAJECTORY_DIR="outputs/musique_official_test500_oracle_step_token_trajectory_10m_20260715_v1"
RETRIEVAL_DIR="outputs/musique_official_test500_oracle_step_token_retrieval_10m_20260715_v1"
ANALYSIS_DIR="outputs/musique_official_test500_state_pointer_analysis_10m_20260715_v1"

# Frozen before this run: dataset-LODO fold that excludes every LongBench MuSiQue query.
SELECTED_HEADS="3:10,21:8,13:6,16:15,14:15,11:3,20:15,8:4,20:6,26:7,16:14,14:14,6:7,14:6,16:13,25:9"
SELECTED_LAYERS="3,6,8,11,13,14,16,20,21,25,26"

if [[ ! -f "${K_PROFILE}/summary.json" ]]; then
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    src/profile_all_head_qk.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir "${CORPUS_DIR}" \
    --profile_dir "${K_PROFILE}" \
    --layers "${SELECTED_LAYERS}" \
    --svd_rank 32 \
    --calibration_blocks 32 \
    --query_vector_tokens 16 \
    --skip_query_profiles \
    --dtype float16 \
    --attn_implementation sdpa \
    --log_every 100
fi

if [[ ! -f "${Q_PROFILE}/step_query_profiles.pt" ]]; then
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    src/profile_step_state_q.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --base_profile_dir "${K_PROFILE}" \
    --step_queries_path "${STEP_PATH}" \
    --output_dir "${Q_PROFILE}" \
    --splits test \
    --task_types multihop \
    --query_vector_tokens 16 \
    --dtype float16 \
    --attn_implementation sdpa
fi

"${PYTHON}" src/convert_step_profiles_to_oracle_trajectory.py \
  --step_profile "${Q_PROFILE}/step_query_profiles.pt" \
  --output_profile "${TRAJECTORY_DIR}/oracle_step_token_q_profiles.pt" \
  --summary_path "${TRAJECTORY_DIR}/summary.json" \
  --mode token_ensemble

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/analyze_generation_retrieval_dynamics.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir "${CORPUS_DIR}" \
  --base_profile_dir "${K_PROFILE}" \
  --trajectory_profile "${TRAJECTORY_DIR}/oracle_step_token_q_profiles.pt" \
  --q_field svd_q \
  --output_dir "${RETRIEVAL_DIR}" \
  --selected_heads "${SELECTED_HEADS}" \
  --head_topk 64 \
  --head_vote_depth 16 \
  --final_blocks 39 \
  --block_batch 32 \
  --skip_bm25

"${PYTHON}" src/analyze_state_pointer_token_routing.py \
  --retrieval_dynamics "${RETRIEVAL_DIR}/retrieval_dynamics.pt" \
  --step_profile "${Q_PROFILE}/step_query_profiles.pt" \
  --output_dir "${ANALYSIS_DIR}" \
  --permutations 20000
