#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
TRACE="${TRACE:-${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260802_qksieve_conditional_value_moments_128k_4gpu}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local layers="$2"
  local label="$3"
  local output_dir="${RUN_ROOT}/${label}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_conditional_value_moments_20260802.py \
    --traces "${TRACE}" \
    --output_dir "${output_dir}" \
    --device cuda \
    --layers "${layers}" \
    --sample_stride 32 \
    --calibration_steps 1 \
    --max_heldout_steps 1 \
    --rate_budget 15 \
    --fractions 0.009765625,0.01953125 \
    --coordinate_dims 4,8,16 \
    --block_sizes 256,512,1024 \
    --moment_bits 4,8 \
    --alphas 0.5,1.0 \
    --ridge 0.01 \
    >"${output_dir}/run.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

run_case 4 0,8 layers_00_08 &
p4=$!
run_case 5 16 layer_16 &
p5=$!
run_case 6 24 layer_24 &
p6=$!
run_case 7 31 layer_31 &
p7=$!

status=0
for pid in "${p4}" "${p5}" "${p6}" "${p7}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
