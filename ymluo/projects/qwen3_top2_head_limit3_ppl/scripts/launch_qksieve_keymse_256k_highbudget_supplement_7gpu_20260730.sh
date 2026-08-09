#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_keymse_256k_highbudget_corrected_c64_7gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
HISTORY_TOKENS="${HISTORY_TOKENS:-262080}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-512}"
VARIANTS="${VARIANTS:-qksieve_deploy_keymse_k20972_c64,qksieve_deploy_keymse_k26214_c64,qksieve_deploy_keymse_k31456_c64}"

mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_window() {
  local gpu_set="$1"
  local topic="$2"
  local seed="$3"
  local output_dir="${RUN_ROOT}/${topic}_seed${seed}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu_set}" \
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
      2>&1 | tee "${output_dir}/run.log"
}

worker_a() {
  run_window "0,1,2" mixed_a 20260732
  run_window "0,1,2" mixed_b 20260733
}

worker_b() {
  run_window "3,4,5,6" mixed_b 20260734
  run_window "3,4,5,6" mixed_a 20260735
}

worker_a >"${RUN_ROOT}/logs/worker_a.log" 2>&1 &
pid_a=$!
worker_b >"${RUN_ROOT}/logs/worker_b.log" 2>&1 &
pid_b=$!

status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
