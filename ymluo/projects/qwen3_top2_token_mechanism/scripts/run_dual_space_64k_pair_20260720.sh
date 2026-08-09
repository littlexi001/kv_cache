#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?GPU index is required}
TOPIC=${2:?topic is required}
ROOT=${3:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/dual_space_64k_pair_20260720}

PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
HEAD_SRC=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src
TOKEN_SRC=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=${HEAD_SRC}:${TOKEN_SRC}
export PATH=/home/fdong/miniconda3/envs/moe/bin:${PATH}
export TORCH_CUDA_ARCH_LIST=8.6

mkdir -p "${ROOT}/${TOPIC}"

full_out="${ROOT}/${TOPIC}/full"
mkdir -p "${full_out}"
${PY} "${HEAD_SRC}/run_head_top2_targeted_ppl_20260714.py" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${full_out}" \
  --topics "${TOPIC}" \
  --window_indices 0 \
  --history_tokens 64000 \
  --query_tokens 32 \
  --eval_tokens 32 \
  --window_stride_tokens 64512 \
  --methods full_attention \
  --full_eval_chunk_tokens 32 \
  --prefill_chunk_tokens 2048 \
  >"${full_out}/run.log" 2>&1

run_sparse() {
  local name=$1
  local mode=$2
  local out="${ROOT}/${TOPIC}/${name}"
  mkdir -p "${out}"
  ${PY} "${HEAD_SRC}/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${out}" \
    --topics "${TOPIC}" \
    --window_indices 0 \
    --history_tokens 64000 \
    --query_tokens 32 \
    --eval_tokens 32 \
    --window_stride_tokens 64512 \
    --mass_thresholds 0.75 \
    --budget_fractions 0.02 \
    --mass_estimator qabs_sampled_tail \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction 0.08 \
    --qabs_use_cuda_kernels \
    --qabs_score_mode "${mode}" \
    --qabs_projection_dim 64 \
    --qabs_value_mass_threshold 1.0 \
    --qabs_partition_ucb_z 0 \
    --qabs_partition_overfetch_factor 2 \
    --qabs_adaptive_rank_energy_threshold 0.85 \
    --qabs_adaptive_rank_residual_precision int4 \
    --qabs_gqa_candidate_mode independent \
    --prefill_chunk_tokens 2048 \
    >"${out}/run.log" 2>&1
}

run_sparse base pca_int4_chunked_logscale16
run_sparse dual_full pca_int4_chunked_logscale16_lowfreq32_int2_union005_refresh4
run_sparse dual_oldest50 pca_int4_chunked_logscale16_lowfreq32_int2_oldest50_union005_refresh4
