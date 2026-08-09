#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_certified_band_refinement_gpu01}"
TRACE32="${TRACE32:-${ROOT}/results/20260726_qk_matrix_spectrum_multimodel_32k/traces/llama31_8b_sports.pt}"
TRACE96="${TRACE96:-${ROOT}/results/20260727_qk_balanced_96k_independent/traces/qwen3_4b_sports96k_32steps.pt}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local label="$2"
  local trace="$3"
  local fraction="$4"
  local output_dir="${RUN_ROOT}/${label}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_certified_band_refinement_20260801.py \
    --trace_path "${trace}" \
    --output_dir "${output_dir}" \
    --label "${label}" \
    --device cuda \
    --layers 0,8,16,24,31 \
    --max_heldout_steps 4 \
    --base_rate_budgets 5,7,9,11 \
    --selected_fractions "${fraction}" \
    --bound_modes global,bandwise \
    --norm_bits 4,8 \
    --norm_block_sizes 256,1024 \
    >"${output_dir}/run.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

run_case 0 llama31_8b_32k "${TRACE32}" 0.04 &
pid0=$!
# The Qwen trace uses layers 0,8,17,26,35, so override the common helper.
(
  output_dir="${RUN_ROOT}/qwen3_4b_96k"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -u \
    src/analyze_qksieve_certified_band_refinement_20260801.py \
    --trace_path "${TRACE96}" \
    --output_dir "${output_dir}" \
    --label qwen3_4b_96k \
    --device cuda \
    --layers 0,8,17,26,35 \
    --max_heldout_steps 4 \
    --base_rate_budgets 5,7,9,11 \
    --selected_fractions 0.013333333333333334 \
    --bound_modes global,bandwise \
    --norm_bits 4,8 \
    --norm_block_sizes 256,1024 \
    >"${output_dir}/run.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
) &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
