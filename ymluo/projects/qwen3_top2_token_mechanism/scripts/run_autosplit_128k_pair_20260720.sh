#!/usr/bin/env bash
set -euo pipefail

GPU_LIST=${1:?"usage: $0 GPU_LIST TOPIC [EVAL_TOKENS] [OUTPUT_ROOT]"}
TOPIC=${2:?"usage: $0 GPU_LIST TOPIC [EVAL_TOKENS] [OUTPUT_ROOT]"}
EVAL_TOKENS=${3:-1024}
OUTPUT_ROOT=${4:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/autosplit_128k_pair_20260720}

PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}
HEAD_SRC=${HEAD_SRC:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src}
MECH_SRC=${MECH_SRC:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src}
DATASET_CACHE=${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}
export PYTHONPATH="${HEAD_SRC}:${MECH_SRC}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU_LIST}

"${PYTHON}" "${HEAD_SRC}/run_adaptive_mass_budget_ppl_20260715.py" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTPUT_ROOT}/autosplit_${TOPIC}_m${EVAL_TOKENS}" \
  --topics "${TOPIC}" \
  --window_indices 0 \
  --history_tokens 128000 \
  --query_tokens "${EVAL_TOKENS}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --window_stride_tokens 130000 \
  --mass_thresholds 0.75 \
  --budget_fractions 0.02 \
  --mass_estimator qabs_sampled_tail \
  --sample_fraction 0.0025 \
  --qabs_dim_count 8 \
  --candidate_fraction 0.08 \
  --qabs_use_cuda_kernels \
  --qabs_score_mode pca_int4_chunked_logscale16_autosplit \
  --qabs_projection_dim 64 \
  --prefill_chunk_tokens 2048 \
  --dataset_cache_dir "${DATASET_CACHE}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced

"${PYTHON}" "${HEAD_SRC}/run_head_top2_targeted_ppl_20260714.py" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTPUT_ROOT}/full_${TOPIC}_m${EVAL_TOKENS}" \
  --topics "${TOPIC}" \
  --window_indices 0 \
  --history_tokens 128000 \
  --query_tokens "${EVAL_TOKENS}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --window_stride_tokens 130000 \
  --top_fraction 0.02 \
  --prefill_chunk_tokens 1024 \
  --full_eval_chunk_tokens 1 \
  --dataset_cache_dir "${DATASET_CACHE}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --methods full_attention
