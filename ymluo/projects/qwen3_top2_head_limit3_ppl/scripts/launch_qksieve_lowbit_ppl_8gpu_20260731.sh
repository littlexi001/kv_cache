#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260731_qksieve_lowbit_ppl_8gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-1024}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_fixedalloc_b12_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_b10_fulltopk_k1280,qksieve_keymse_requestlocal_fixedalloc_b8_fulltopk_k1280}"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

jobs=(
  "0|sports|20260741"
  "1|medicine|20260742"
  "2|computer|20260743"
  "3|space|20260744"
  "4|politics|20260745"
  "5|religion|20260746"
  "6|mixed_a|20260747"
  "7|mixed_b|20260748"
)

pids=()
for job in "${jobs[@]}"; do
  IFS="|" read -r gpu topic seed <<<"${job}"
  output_dir="${RUN_ROOT}/${topic}_seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    echo "SKIP completed: ${topic}_seed${seed}"
    continue
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
      --model_name_or_path "${MODEL}" \
      --template "${TEMPLATE}" \
      --output_dir "${output_dir}" \
      --history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --topic "${topic}" \
      --prefill_chunk_tokens "${PREFILL_CHUNK_TOKENS}" \
      --dataset_cache_dir "${DATASET_CACHE_DIR}" \
      --seed "${seed}" \
      --dtype float16 \
      --device cuda \
      --device_map balanced \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANTS}" \
      >"${output_dir}/run.log" 2>&1
  ) >"${RUN_ROOT}/logs/${topic}_seed${seed}.log" 2>&1 &
  pids+=("$!")
  echo "${topic}: GPU ${gpu}, PID $!"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
