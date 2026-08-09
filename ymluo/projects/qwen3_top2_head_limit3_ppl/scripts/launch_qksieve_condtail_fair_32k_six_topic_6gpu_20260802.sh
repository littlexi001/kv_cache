#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
DATASET_CACHE="${DATASET_CACHE:-/home/fdong/ymluo/datasets/sklearn}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260802_qksieve_condtail_fair_32k_six_topic_6gpu}"

VARIANTS="qksieve_keymse_requestlocal_sampled_k1280_c32,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280,qksieve_keymse_requestlocal_condtail8_k1280_c32"
TOPICS=(sports medicine mixed_b computer politics religion)

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

pids=()
for gpu in 0 1 2 3 4 5; do
  topic="${TOPICS[$gpu]}"
  out="${RUN_ROOT}/${topic}"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${out}" \
    --history_tokens 32768 \
    --eval_tokens 64 \
    --topic "${topic}" \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE}" \
    --seed 20260831 \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --repeat_topic_stream_if_short \
    --variants "${VARIANTS}" \
    >"${RUN_ROOT}/logs/${topic}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done

if [[ "${status}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit "${status}"
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
