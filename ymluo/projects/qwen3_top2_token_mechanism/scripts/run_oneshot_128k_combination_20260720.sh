#!/usr/bin/env bash
set -euo pipefail

GPU_LIST=${1:?"usage: $0 GPU_LIST METHOD TOPIC [EVAL_TOKENS] [OUTPUT_ROOT]"}
METHOD=${2:?"method must be fixed2 or budget"}
TOPIC=${3:?"topic is required"}
EVAL_TOKENS=${4:-1024}
OUTPUT_ROOT=${5:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/oneshot_128k_combinations_20260720}

case "${METHOD}" in
  fixed2)
    SCORE_MODE=pca_int4_logscale16_oneshot95_fixed2_autosplit
    BUDGETS=0.02
    UCB_Z=1.64
    OVERFETCH=0
    ;;
  budget)
    SCORE_MODE=pca_int4_logscale16_oneshot95_budget_autosplit
    BUDGETS=0.005,0.01,0.02,0.03,0.04,0.06,0.08
    UCB_Z=0.0
    OVERFETCH=2
    ;;
  *)
    echo "unknown method: ${METHOD}" >&2
    exit 2
    ;;
esac

PYTHON=${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}
HEAD_SRC=${HEAD_SRC:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src}
MECH_SRC=${MECH_SRC:-/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/src}
DATASET_CACHE=${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}
export PYTHONPATH="${HEAD_SRC}:${MECH_SRC}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${GPU_LIST}

OUTPUT_DIR=${OUTPUT_ROOT}/${METHOD}_${TOPIC}_m${EVAL_TOKENS}
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
  --budget_fractions "${BUDGETS}" \
  --mass_estimator qabs_sampled_tail \
  --sample_fraction 0.0025 \
  --qabs_dim_count 8 \
  --candidate_fraction 0.08 \
  --qabs_use_cuda_kernels \
  --qabs_score_mode "${SCORE_MODE}" \
  --qabs_projection_dim 64 \
  --qabs_partition_ucb_z "${UCB_Z}" \
  --qabs_partition_overfetch_factor "${OVERFETCH}" \
  --qabs_early_layer_count 0 \
  --qabs_gqa_candidate_mode independent \
  --prefill_chunk_tokens 2048 \
  --dataset_cache_dir "${DATASET_CACHE}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced
