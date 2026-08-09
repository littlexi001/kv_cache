#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/experiments/frozen_c64_20260807/src}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/qksieve_fier_budget_isolated_64k_20260808}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
DRIVER="${SRC_ROOT}/run_qksieve_fier_budget_ab_20260808.py"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
HISTORY_TOKENS="${HISTORY_TOKENS:-65536}"
EVAL_TOKENS="${EVAL_TOKENS:-64}"
ROUNDS="${ROUNDS:-3}"

VARIANTS=(
  qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280
  qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k1280
  qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k512
  qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k512
)

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=0
export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1
export QKSIEVE_PARALLEL_QK_WORKERS="${QKSIEVE_PARALLEL_QK_WORKERS:-12}"
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
unset QKSIEVE_PROFILE_STAGES || true

mkdir -p "${RUN_ROOT}/logs"

run_one() {
  local round="$1"
  local index="$2"
  local variant="${VARIANTS[index]}"
  local gpu="$(( (index + round - 1) % 4 ))"
  local output_dir="${RUN_ROOT}/round${round}/${variant}"
  local log="${RUN_ROOT}/logs/round${round}_${variant}.log"
  if [[ -f "${output_dir}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${DRIVER}" \
    --model_name_or_path "${MODEL}" \
    --template "${ROOT}/nonexistent_template.pt" \
    --output_dir "${output_dir}" \
    --history_tokens "${HISTORY_TOKENS}" \
    --stream_reference_history_tokens "${HISTORY_TOKENS}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --text_file "${TEXT_FILE}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${ROOT}/cache/sklearn" \
    --seed "$((20260808 + round))" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "${variant}" \
    >"${log}" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
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

"${PYTHON}" "${ROOT}/src/summarize_qksieve_fier_budget_isolated_20260808.py" \
  "${RUN_ROOT}"
touch "${RUN_ROOT}/ALL_COMPLETE"
