#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massfloor_closedloop_small_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="qksieve_keymse_requestlocal_valuesketch16i4_wometric_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor900_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor950_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor975_fulltopk_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_32k() {
  local gpu="$1"
  local topic="$2"
  local seed="$3"
  local output_dir="${OUTPUT}/${topic}32k"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${QWEN_MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens 32000 \
    --stream_reference_history_tokens 32000 \
    --eval_tokens 8 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${OUTPUT}/logs/${topic}32k.log" 2>&1
  touch "${OUTPUT}/${topic}32k_COMPLETE"
}

run_128k() {
  local gpu_pair="$1"
  local topic="$2"
  local seed="$3"
  local output_dir="${OUTPUT}/${topic}128k"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_pair}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${LLAMA_MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens 131008 \
    --stream_reference_history_tokens 131008 \
    --eval_tokens 4 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    >"${OUTPUT}/logs/${topic}128k.log" 2>&1
  touch "${OUTPUT}/${topic}128k_COMPLETE"
}

run_32k 1 sports 20260831 & pid1=$!
run_32k 2 medicine 20260832 & pid2=$!
run_128k 3,4 computer 20260833 & pid34=$!

wait "${pid1}"
wait "${pid2}"
wait "${pid34}"
touch "${OUTPUT}/ALL_COMPLETE"
