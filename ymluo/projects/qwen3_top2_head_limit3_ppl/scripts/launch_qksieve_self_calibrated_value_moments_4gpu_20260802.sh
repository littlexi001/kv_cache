#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260802_qksieve_self_calibrated_value_moments_4gpu}"
TRACE_ROOT="${TRACE_ROOT:-${ROOT}/results/20260717_delta_qkv_traces_32k_s16}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local topic="$2"
  local layers="$3"
  local label="$4"
  local output_dir="${RUN_ROOT}/${label}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_self_calibrated_value_moments_20260802.py \
    --traces "${TRACE_ROOT}/${topic}.pt" \
    --output_dir "${output_dir}" \
    --device cuda \
    --layers "${layers}" \
    --sample_stride 32 \
    --calibration_steps 8 \
    --max_heldout_steps 8 \
    --rate_budget 15 \
    --fraction 0.04 \
    --ridge 0.01 \
    --selection_mode oracle_regret \
    --tolerances 0.0025,0.005,0.01,0.02,0.05 \
    >"${output_dir}/run.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

run_case "${GPU0:-0}" sports 0,8,16 sports_early &
p0=$!
run_case "${GPU1:-1}" sports 24,31 sports_late &
p1=$!
run_case "${GPU2:-2}" medicine 0,8,16 medicine_early &
p2=$!
run_case "${GPU3:-3}" medicine 24,31 medicine_late &
p3=$!

status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
