#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_native128k_blockmass_cross_topic_gpu0123}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="qksieve_keymse_requestlocal_blockmass32_fulltopk_k1280,qksieve_keymse_requestlocal_blockmass64_fulltopk_k1280,qksieve_keymse_requestlocal_blockmass128_fulltopk_k1280"
WORKERS=(
  "0,1|computer|20260833|mixed_b|20260836"
  "2,3|sports_both|20260831|medicine|20260832|religion|20260835"
)

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

pids=()
for worker in "${WORKERS[@]}"; do
  IFS="|" read -r -a fields <<<"${worker}"
  gpus="${fields[0]}"
  (
    for ((index=1; index<${#fields[@]}; index+=2)); do
      topic="${fields[$index]}"
      seed="${fields[$((index+1))]}"
      output_dir="${RUN_ROOT}/${topic}_seed${seed}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
        echo "SKIP ${topic}/seed${seed}"
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
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
        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --max_memory_per_gpu_gib 22 \
        --variants "${VARIANTS}" \
        >"${output_dir}/run.log" 2>&1
      touch "${output_dir}/ALL_COMPLETE"
    done
  ) >"${RUN_ROOT}/logs/gpu_${gpus//,/_}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
