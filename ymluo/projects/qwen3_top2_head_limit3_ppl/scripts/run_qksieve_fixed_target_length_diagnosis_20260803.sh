#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT must name a result directory}"
GPU_IDS="${GPU_IDS:?GPU_IDS must contain a physical GPU pair, for example 0,1}"
TOPIC="${TOPIC:?TOPIC is required}"
SEED="${SEED:?SEED is required}"
HISTORY_TOKENS="${HISTORY_TOKENS:-65472 81856 98240 114624 131008}"
REFERENCE_HISTORY_TOKENS="${REFERENCE_HISTORY_TOKENS:-131008}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="${VARIANTS:-exact_qk_oracle_k1280,qksieve_keymse_requestlocal_fulltopk_k1280,qksieve_keymse_requestlocal_sampled_k1280_c32,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280}"
COLLECT_QK_PRODUCT_SPECTRUM="${COLLECT_QK_PRODUCT_SPECTRUM:-1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

extra_args=()
if [[ "${COLLECT_QK_PRODUCT_SPECTRUM}" == "1" ]]; then
  extra_args+=(--collect_qk_product_spectrum)
fi

for history_tokens in ${HISTORY_TOKENS}; do
  output_dir="${RUN_ROOT}/${TOPIC}_seed${SEED}_h${history_tokens}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    continue
  fi

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens "${history_tokens}" \
    --stream_reference_history_tokens "${REFERENCE_HISTORY_TOKENS}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --topic "${TOPIC}" \
    --seed "${SEED}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    "${extra_args[@]}" \
    >"${RUN_ROOT}/logs/${TOPIC}_h${history_tokens}.log" 2>&1
done

touch "${RUN_ROOT}/${TOPIC}_ALL_COMPLETE"
