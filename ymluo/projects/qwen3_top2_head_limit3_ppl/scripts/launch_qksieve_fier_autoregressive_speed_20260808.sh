#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/experiments/frozen_c64_20260807/src}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/qksieve_fier_autoregressive_64k_20260808}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
ROUNDS="${ROUNDS:-3}"
GENERATION_STEPS="${GENERATION_STEPS:-256}"

METHODS=(
  full
  qksieve_no_value_top1280
  fier_rtn1_g32_top1280
  fier_rtn1_g32_top512
)

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=0
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=0
export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1
export QKSIEVE_PARALLEL_QK_WORKERS=12
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FIER_ATTENTION_SPLIT_OVERRIDE=8
unset QKSIEVE_PROFILE_STAGES || true

mkdir -p "${RUN_ROOT}/logs"

run_one() {
  local round="$1"
  local index="$2"
  local method="${METHODS[index]}"
  local gpu="$(( (index + round - 1) % 4 ))"
  local output_dir="${RUN_ROOT}/round${round}"
  local output="${output_dir}/${method}.json"
  local log="${RUN_ROOT}/logs/round${round}_${method}.log"
  if [[ -f "${output}" ]]; then
    return
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    "${SRC_ROOT}/run_qksieve_fier_autoregressive_speed_20260808.py" \
    --model_name_or_path "${MODEL}" \
    --text_file "${TEXT_FILE}" \
    --output "${output}" \
    --method "${method}" \
    --history_tokens 65536 \
    --generation_steps "${GENERATION_STEPS}" \
    --steady_start 16 \
    --prefill_chunk_tokens 1024 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --seed "$((20260808 + round))" \
    >"${log}" 2>&1
}

for round in $(seq 1 "${ROUNDS}"); do
  pids=()
  for index in 0 1 2 3; do
    run_one "${round}" "${index}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

"${PYTHON}" "${SRC_ROOT}/summarize_qksieve_fier_autoregressive_speed_20260808.py" \
  "${RUN_ROOT}" | tee "${RUN_ROOT}/summary.log"
touch "${RUN_ROOT}/ALL_COMPLETE"
