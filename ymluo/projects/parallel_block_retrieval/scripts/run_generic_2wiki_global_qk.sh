#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-1,3,5}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  src/run_global_step_block_retrieval.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --corpus_dir data/real_longbench_docqa_10m_clean_record64 \
  --profile_dir outputs/real_longbench_docqa_10m_prerope_qk64_question16_profile \
  --step_queries_path data/real_2wiki_generic_steps_v1/step_queries.jsonl \
  --output_dir outputs/real_2wiki_generic_global_qk_v1 \
  --splits train,dev,test \
  --task_types multihop \
  --exclude_query_ids '' \
  --svd_rank 32 \
  --candidate_blocks 512 \
  --target_blocks 16 \
  --query_tokens 16
