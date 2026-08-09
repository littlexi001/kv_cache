#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_keymse_256k_highbudget_frontier_7gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
GPU_SET="${GPU_SET:-0,1,2,3,4,5,6}"
SEED="${SEED:-20260731}"
HISTORY_TOKENS="${HISTORY_TOKENS:-262080}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1024}"
VARIANTS="${VARIANTS:-qksieve_keymse_fulltopk_k7680,qksieve_deploy_keymse_k7680,qksieve_keymse_fulltopk_k10240,qksieve_deploy_keymse_k10240,qksieve_keymse_fulltopk_k12800,qksieve_deploy_keymse_k12800,qksieve_keymse_fulltopk_k15728,qksieve_deploy_keymse_k15728}"

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${RUN_ROOT}" \
  --history_tokens "${HISTORY_TOKENS}" \
  --eval_tokens "${EVAL_TOKENS}" \
  --topic mixed_b \
  --prefill_chunk_tokens "${PREFILL_CHUNK_TOKENS}" \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --seed "${SEED}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANTS}" \
  2>&1 | tee "${RUN_ROOT}/run.log"
