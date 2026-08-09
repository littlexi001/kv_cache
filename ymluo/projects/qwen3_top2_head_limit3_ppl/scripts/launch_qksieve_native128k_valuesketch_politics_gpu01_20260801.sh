#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_native128k_valuesketch_politics_gpu01}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch8i4_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch32i4_fulltopk_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${RUN_ROOT}/politics_seed20260834" \
  --history_tokens 131008 \
  --eval_tokens 64 \
  --topic politics \
  --seed 20260834 \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANTS}" \
  >"${RUN_ROOT}/run.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
