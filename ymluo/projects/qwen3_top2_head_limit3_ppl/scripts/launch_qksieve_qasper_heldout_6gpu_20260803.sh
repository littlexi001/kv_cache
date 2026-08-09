#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_qasper_heldout_rate19_rank16_6gpu_v1}"
KEY_RATE_BUDGET="${KEY_RATE_BUDGET:-19}"
VALUE_RANK="${VALUE_RANK:-16}"
QUERY_FACTOR_SOURCE="${QUERY_FACTOR_SOURCE:-prefill}"
QUERY_FACTOR_PREFILL_TOKENS="${QUERY_FACTOR_PREFILL_TOKENS:-8}"
KEY_QUANTIZER="${KEY_QUANTIZER:-metric}"
KEY_ALLOCATION_OBJECTIVE="${KEY_ALLOCATION_OBJECTIVE:-key_mse}"
COVERAGE_TARGETS="${COVERAGE_TARGETS:-0.95}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"
mapfile -t TRACES < <(find "${TRACE_ROOT}" -maxdepth 1 -type f -name '*.pt' | sort)
if [[ "${#TRACES[@]}" -eq 0 ]]; then
  echo "no held-out traces found under ${TRACE_ROOT}" >&2
  exit 1
fi

run_worker() {
  local gpu="$1"
  local index
  for ((index=gpu; index<${#TRACES[@]}; index+=6)); do
    local trace="${TRACES[index]}"
    local name
    name="$(basename "${trace}" .pt)"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
      src/analyze_qksieve_output_risk_budget_20260803.py \
      --trace "${trace}" \
      --output_dir "${OUTPUT}/${name}" \
      --model_name_or_path "${MODEL}" \
      --device cuda \
      --fixed_top_k 1280 \
      --coverage_targets "${COVERAGE_TARGETS}" \
      --coverage_histogram_bins 256 \
      --minimum_top_k 256 \
      --maximum_top_k 0 \
      --key_rate_budget "${KEY_RATE_BUDGET}" \
      --key_quantizer "${KEY_QUANTIZER}" \
      --key_allocation_objective "${KEY_ALLOCATION_OBJECTIVE}" \
      --value_rank "${VALUE_RANK}" \
      --value_bits 4 \
      --risk_bits 4 \
      --query_factor_source "${QUERY_FACTOR_SOURCE}" \
      --query_factor_prefill_tokens "${QUERY_FACTOR_PREFILL_TOKENS}" \
      --focus_mass_ladder \
      >"${OUTPUT}/logs/${name}.log" 2>&1
    touch "${OUTPUT}/${name}_COMPLETE"
  done
}

pids=()
for gpu in 0 1 2 3 4 5; do
  run_worker "${gpu}" &
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
