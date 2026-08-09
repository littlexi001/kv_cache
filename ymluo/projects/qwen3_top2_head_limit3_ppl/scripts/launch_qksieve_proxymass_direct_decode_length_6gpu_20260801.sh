#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_proxymass_direct_decode_length_6gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

workers=(
  "0|8192"
  "1|16384"
  "2,3|32768,65536"
  "4,5|131040"
)
pids=()
for worker in "${workers[@]}"; do
  IFS="|" read -r gpus lengths <<<"${worker}"
  (
    IFS="," read -r -a length_values <<<"${lengths}"
    for length in "${length_values[@]}"; do
      if (( length <= 8192 )); then
        budget=492
      elif (( length <= 16384 )); then
        budget=984
      else
        budget=1280
      fi
      output_dir="${RUN_ROOT}/n${length}"
      mkdir -p "${output_dir}"
      if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
        echo "SKIP n${length}"
        continue
      fi
      variant="qksieve_keymse_requestlocal_proxymass_k${budget}_c32"
      CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
        src/run_qksieve_coldskip_longcontext_quality_20260730.py \
        --model_name_or_path "${MODEL}" \
        --template "${TEMPLATE}" \
        --output_dir "${output_dir}" \
        --history_tokens "${length}" \
        --eval_tokens "${EVAL_TOKENS}" \
        --topic mixed_b \
        --seed "$((20260840 + length / 8192))" \
        --repeat_topic_stream_if_short \
        --prefill_chunk_tokens 1024 \
        --dataset_cache_dir "${DATASET_CACHE_DIR}" \
        --dtype float16 \
        --device cuda \
        --device_map balanced \
        --max_memory_per_gpu_gib 22 \
        --variants "${variant}" \
        >"${output_dir}/run.log" 2>&1
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
