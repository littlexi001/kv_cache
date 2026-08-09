#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_nested_overfetch_refinement_gpu01}"
TRACE32="${TRACE32:-${ROOT}/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt}"
TRACE96="${TRACE96:-${ROOT}/results/20260727_qk_balanced_96k_independent/traces/qwen3_4b_sports96k_32steps.pt}"
GPU32="${GPU32:-0}"
GPU96="${GPU96:-1}"
CASE_SET="${CASE_SET:-all}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local layers="$4"
  local fraction="$5"
  local output_dir="${RUN_ROOT}/${label}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_nested_overfetch_refinement_20260801.py \
    --trace_path "${trace}" \
    --output_dir "${output_dir}" \
    --label "${label}" \
    --device cuda \
    --layers "${layers}" \
    --max_heldout_steps 8 \
    --base_rate_budgets 5,7,9,11 \
    --selected_fractions "${fraction}" \
    --overfetch_factors 1.5,2,3,4,6,8 \
    >"${output_dir}/run.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

status=0
pids=()
if [[ "${CASE_SET}" == "all" || "${CASE_SET}" == "32" ]]; then
  run_case "${GPU32}" llama31_8b_32k "${TRACE32}" 0,8,16,24,31 0.04 &
  pids+=("$!")
fi
if [[ "${CASE_SET}" == "all" || "${CASE_SET}" == "96" ]]; then
  run_case "${GPU96}" qwen3_4b_96k "${TRACE96}" 0,8,17,26,35 0.013333333333333334 &
  pids+=("$!")
fi
if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "CASE_SET must be one of: all, 32, 96" >&2
  exit 2
fi
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
