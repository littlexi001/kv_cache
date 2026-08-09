#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_valuesketch32_alpha_computer_medicine128k_6gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANT="qksieve_keymse_requestlocal_valuesketch32i4_sampled_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
cd "${ROOT}"
mkdir -p "${RUN_ROOT}/logs"

run_case() {
  local gpu_ids="$1"
  local topic="$2"
  local seed="$3"
  local alpha="$4"
  local alpha_label="${alpha/./p}"
  local output_dir="${RUN_ROOT}/${topic}_alpha${alpha_label}_seed${seed}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  QKSIEVE_VALUE_SKETCH_TAIL_ALPHA="${alpha}" \
  "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens 131008 \
    --eval_tokens 64 \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANT}" \
    >"${RUN_ROOT}/logs/${topic}_alpha${alpha_label}.log" 2>&1
}

worker() {
  local gpu_ids="$1"
  local alpha="$2"
  run_case "${gpu_ids}" computer 20260833 "${alpha}"
  run_case "${gpu_ids}" medicine 20260832 "${alpha}"
}

worker 0,1 0.25 & p0=$!
worker 2,3 0.50 & p1=$!
worker 4,5 0.75 & p2=$!

status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
wait "${p2}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
