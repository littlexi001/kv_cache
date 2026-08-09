#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_qkmse_qwen96k_2gpu_v1}"
GPU_BASE="${GPU_BASE:-4}"
KEY_ALLOCATION_OBJECTIVE="${KEY_ALLOCATION_OBJECTIVE:-qk_mse}"
COVERAGE_TARGETS="${COVERAGE_TARGETS:-0.95}"
KEY_RATE_BUDGET="${KEY_RATE_BUDGET:-19}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

topics=(sports medicine)
pids=()
for local_index in 0 1; do
  gpu=$((local_index + GPU_BASE))
  topic="${topics[local_index]}"
  destination="${OUTPUT}/${topic}_${KEY_ALLOCATION_OBJECTIVE}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
      src/analyze_qksieve_output_risk_budget_20260803.py \
      --trace "${TRACE_ROOT}/${topic}.pt" \
      --output_dir "${destination}" \
      --model_name_or_path "${MODEL}" \
      --device cuda \
      --fixed_top_k 1280 \
      --coverage_targets "${COVERAGE_TARGETS}" \
      --coverage_histogram_bins 256 \
      --minimum_top_k 256 \
      --maximum_top_k 0 \
      --key_rate_budget "${KEY_RATE_BUDGET}" \
      --key_quantizer plain \
      --key_allocation_objective "${KEY_ALLOCATION_OBJECTIVE}" \
      --value_rank 16 \
      --value_bits 4 \
      --risk_bits 4 \
      --query_factor_source prefill \
      --query_factor_prefill_tokens 8 \
      --focus_mass_ladder \
      >"${OUTPUT}/logs/${topic}.log" 2>&1
    touch "${destination}_COMPLETE"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
