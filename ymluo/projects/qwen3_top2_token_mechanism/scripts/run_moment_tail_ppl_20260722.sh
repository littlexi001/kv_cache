#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?'usage: GPU SCORE_MODE TOPIC HISTORY_TOKENS EVAL_TOKENS OUTPUT_DIR'}
SCORE_MODE=${2:?'score mode is required'}
TOPIC=${3:?'topic is required'}
HISTORY_TOKENS=${4:?'history length is required'}
EVAL_TOKENS=${5:?'evaluation length is required'}
OUTPUT_DIR=${6:?'output directory is required'}

PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
HEAD_SRC=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src
MECH_SRC=/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src
DATASET_CACHE=/home/fdong/ymluo/datasets/sklearn

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH="${HEAD_SRC}:${MECH_SRC}:${PYTHONPATH:-}"
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -u "${HEAD_SRC}/run_adaptive_mass_budget_ppl_20260715.py" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --topics "${TOPIC}" \
    --window_indices 0 \
    --history_tokens "${HISTORY_TOKENS}" \
    --query_tokens "${EVAL_TOKENS}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --window_stride_tokens $((HISTORY_TOKENS + EVAL_TOKENS + 1024)) \
    --mass_thresholds 0.75 \
    --budget_fractions 0.02 \
    --mass_estimator qabs_sampled_tail \
    --sample_fraction 0.0025 \
    --qabs_dim_count 8 \
    --candidate_fraction 0.06 \
    --qabs_use_cuda_kernels \
    --qabs_score_mode "${SCORE_MODE}" \
    --qabs_projection_dim 48 \
    --prefill_chunk_tokens 2048 \
    --dataset_cache_dir "${DATASET_CACHE}" \
    --seed 20260714 \
    --dtype float16 \
    --device cuda \
    --device_map auto
