#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_prerope_stage_profile}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
GPU="${GPU:-0}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_post2xprererank_b8_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_post2xboundary75prererank_b8_fulltopk_k1280}"
HISTORY_LENGTHS="${HISTORY_LENGTHS:-32768 65536}"
EVAL_TOKENS="${EVAL_TOKENS:-16}"

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export QKSIEVE_PROFILE_STAGES=1

for history_tokens in ${HISTORY_LENGTHS}; do
  output_dir="${RUN_ROOT}/${history_tokens}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    echo "SKIP completed: ${history_tokens}"
    continue
  fi
  "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens "${history_tokens}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --topic mixed_b \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --seed 20260748 \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${output_dir}/run.log" 2>&1
done

touch "${RUN_ROOT}/ALL_COMPLETE"
