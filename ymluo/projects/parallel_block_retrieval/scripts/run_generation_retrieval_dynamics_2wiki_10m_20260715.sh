#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

TRAJECTORY_DIR="outputs/real_2wiki_generation_q_trajectory_20260715_v1"
RESULT_DIR="outputs/real_2wiki_generation_retrieval_dynamics_10m_20260715_v1"
BASE_PROFILE="outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/profile_generation_q_trajectory.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --step_queries_path data/real_2wiki_generic_steps_v1/step_queries.jsonl \
  --base_profile_dir "${BASE_PROFILE}" \
  --output_dir "${TRAJECTORY_DIR}" \
  --splits train,dev,test \
  --max_new_tokens 24 \
  --prompt_mode bridge_reasoned \
  --dtype float16

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/analyze_generation_retrieval_dynamics.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --base_profile_dir "${BASE_PROFILE}" \
  --trajectory_profile "${TRAJECTORY_DIR}/trajectory_q_profiles.pt" \
  --output_dir "${RESULT_DIR}" \
  --head_topk 64 \
  --head_vote_depth 16 \
  --final_blocks 39 \
  --block_batch 32
