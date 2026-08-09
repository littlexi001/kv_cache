#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/parallel_block_retrieval}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${GPUS:-0,7}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

BASE_PROFILE="outputs/real_longbench_docqa_10m_allhead_prerope_svd32_profile"
TRAJECTORY_PROFILE="outputs/real_2wiki_smoothed_q_trajectory_20260715_v1/trajectory_smoothed_q_profiles.pt"
CORPUS_DIR="data/real_longbench_docqa_10m_clean_record64"

FIELDS=(
  native_ema_w2
  native_ema_w4
  native_ema_w8
  native_evidence_probe_q_mix_n25
  native_evidence_probe_q_mix_n50
  native_evidence_probe_q_mix_n75
)

for field in "${FIELDS[@]}"; do
  output_dir="outputs/real_2wiki_${field}_retrieval_dynamics_10m_20260715_v1"
  "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC}" \
    src/analyze_generation_retrieval_dynamics.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --corpus_dir "${CORPUS_DIR}" \
    --base_profile_dir "${BASE_PROFILE}" \
    --trajectory_profile "${TRAJECTORY_PROFILE}" \
    --q_field "${field}" \
    --output_dir "${output_dir}" \
    --head_topk 64 \
    --head_vote_depth 16 \
    --final_blocks 39 \
    --block_batch 32 \
    --skip_bm25
done
