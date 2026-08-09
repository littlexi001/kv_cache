#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/experiments/frozen_c64_20260807/src}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/fier_split16_64k_20260808}"
BASELINE_ROOT="${BASELINE_ROOT:-${ROOT}/results/qksieve_fier_budget_isolated_64k_20260808}"
SPLIT="${SPLIT:-16}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
DRIVER="${SRC_ROOT}/run_qksieve_fier_budget_ab_20260808.py"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
VARIANT="qksieve_qmse_requestlocal_fier_rtn1_g32_fulltopk_k512"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${SRC_ROOT}:${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=0
export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1
export QKSIEVE_FIER_ATTENTION_SPLIT_OVERRIDE="${SPLIT}"
unset QKSIEVE_PROFILE_STAGES || true

mkdir -p "${RUN_ROOT}/logs"
run_one() {
  local round="$1"
  local gpu="$2"
  local output="${RUN_ROOT}/round${round}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${DRIVER}" \
    --model_name_or_path "${MODEL}" \
    --template "${ROOT}/nonexistent_template.pt" \
    --output_dir "${output}" \
    --history_tokens 65536 \
    --stream_reference_history_tokens 65536 \
    --eval_tokens 64 \
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
    --variants "${VARIANT}" \
    >"${RUN_ROOT}/logs/round${round}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

run_one 1 2 & pid1=$!
run_one 2 3 & pid2=$!
run_one 3 0 & pid3=$!
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

"${PYTHON}" "${ROOT}/src/summarize_fier_split16_20260808.py" \
  "${BASELINE_ROOT}" "${RUN_ROOT}" "${SPLIT}"
touch "${RUN_ROOT}/ALL_COMPLETE"
