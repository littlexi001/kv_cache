#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must name a fresh result directory}"
GPU_IDS="${GPU_IDS:?GPU_IDS must contain a physical GPU pair, for example 3,4}"
TOPICS="${TOPICS:?TOPICS must contain space-separated topic:seed entries}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch8i4_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_fulltopk_k1280}"
PROTECT_RECENT_TOKENS="${PROTECT_RECENT_TOKENS:-0}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

for entry in ${TOPICS}; do
  topic="${entry%%:*}"
  seed="${entry##*:}"
  output_dir="${RUN_ROOT}/${topic}_seed${seed}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    continue
  fi

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens 131008 \
    --eval_tokens 64 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens "${PROTECT_RECENT_TOKENS}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${RUN_ROOT}/logs/${topic}.log" 2>&1
done

touch "${RUN_ROOT}/ALL_COMPLETE"
