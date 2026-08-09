#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_qkmse_qwen32k_4gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

topics=(sports sports medicine medicine)
objectives=(key_mse qk_mse key_mse qk_mse)
pids=()
for gpu in 0 1 2 3; do
  topic="${topics[gpu]}"
  objective="${objectives[gpu]}"
  name="${topic}_${objective}"
  destination="${OUTPUT}/${name}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
      src/analyze_qksieve_output_risk_budget_20260803.py \
      --trace "${TRACE_ROOT}/${topic}.pt" \
      --output_dir "${destination}" \
      --model_name_or_path "${MODEL}" \
      --device cuda \
      --fixed_top_k 1280 \
      --coverage_targets 0.95 \
      --coverage_histogram_bins 256 \
      --minimum_top_k 256 \
      --maximum_top_k 0 \
      --key_rate_budget 19 \
      --key_quantizer plain \
      --key_allocation_objective "${objective}" \
      --value_rank 16 \
      --value_bits 4 \
      --risk_bits 4 \
      --query_factor_source prefill \
      --query_factor_prefill_tokens 8 \
      --focus_mass_ladder \
      >"${OUTPUT}/logs/${name}.log" 2>&1
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
