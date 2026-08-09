#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TEMPLATE="${TEMPLATE:-${PROJECT_ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/results/20260730_qksieve_keymse_512k_extrap_budget_frontier_7gpu}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
GPU_SET="${GPU_SET:-0,1,2,3,4,5,6}"
SEED="${SEED:-20260731}"
HISTORY_TOKENS="${HISTORY_TOKENS:-524256}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
PREFILL_CHUNKS="${PREFILL_CHUNKS:-512 256}"
VARIANTS="${VARIANTS:-qksieve_keymse_fulltopk_k15728,qksieve_deploy_keymse_k15728,qksieve_keymse_fulltopk_k20972,qksieve_deploy_keymse_k20972,qksieve_keymse_fulltopk_k26214,qksieve_deploy_keymse_k26214,qksieve_keymse_fulltopk_k31456,qksieve_deploy_keymse_k31456}"

mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export PATH="/home/fdong/miniconda3/envs/moe/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_SET}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for chunk in ${PREFILL_CHUNKS}; do
  output_dir="${RUN_ROOT}/chunk${chunk}"
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    touch "${RUN_ROOT}/ALL_COMPLETE"
    exit 0
  fi
  set +e
  "${PYTHON}" -u src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens "${HISTORY_TOKENS}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --topic mixed_b \
    --prefill_chunk_tokens "${chunk}" \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --seed "${SEED}" \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANTS}" \
    --allow_context_extrapolation \
    2>&1 | tee "${output_dir}/run.log"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -eq 0 && -f "${output_dir}/ALL_COMPLETE" ]]; then
    touch "${RUN_ROOT}/ALL_COMPLETE"
    exit 0
  fi
  echo "[retry] chunk=${chunk} exited ${status}" \
    | tee -a "${RUN_ROOT}/retry.log"
  if ! grep -Eiq \
    "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory" \
    "${output_dir}/run.log"; then
    echo "[stop] failure is not an OOM; changing prefill chunk cannot fix it" \
      | tee -a "${RUN_ROOT}/retry.log"
    exit "${status}"
  fi
done

echo "All configured 512K retries failed." >&2
exit 1
