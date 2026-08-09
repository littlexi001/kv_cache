#!/usr/bin/env bash
set -euo pipefail

GPU_LIST=${1:?'usage: GPU_LIST TOPIC [EVAL_TOKENS] [OUTPUT_ROOT] [RANK] [CANDIDATE_FRACTION]'}
TOPIC=${2:?'topic is required'}
EVAL_TOKENS=${3:-1024}
OUTPUT_ROOT=${4:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/qkmetric_128k_20260721}
RANK=${5:-64}
CANDIDATE_FRACTION=${6:-0.06}

PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}
HEAD_SRC=${HEAD_SRC:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src}
MECH_SRC=${MECH_SRC:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src}
DATASET_CACHE=${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH="${HEAD_SRC}:${MECH_SRC}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU_LIST}

CANDIDATE_TAG=$(printf '%s' "${CANDIDATE_FRACTION}" | tr -d '.')
OUTPUT_DIR=${OUTPUT_ROOT}/qkmetric_r${RANK}_candidate${CANDIDATE_TAG}_${TOPIC}_m${EVAL_TOKENS}
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${HEAD_SRC}/run_adaptive_mass_budget_ppl_20260715.py" \
  --model_name_or_path "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
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
  --candidate_fraction "${CANDIDATE_FRACTION}" \
  --qabs_use_cuda_kernels \
  --qabs_score_mode pca_int4_chunked_logscale16_qkmetric_autosplit \
  --qabs_projection_dim "${RANK}" \
  --prefill_chunk_tokens 2048 \
  --dataset_cache_dir "${DATASET_CACHE}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced
