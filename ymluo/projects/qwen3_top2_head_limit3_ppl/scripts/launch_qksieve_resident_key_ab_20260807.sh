#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/resident_key_ab_32k_20260807}"
GPU="${GPU:-0}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
SEED="${SEED:-20260839}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_RESIDENT_VALUE_WORKERS=12
export QKSIEVE_RESIDENT_KEY_WORKERS=36
export QKSIEVE_PARALLEL_QK_WORKERS=36
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
{
  echo "root=${ROOT}"
  echo "model=${MODEL}"
  echo "text_file=${TEXT_FILE}"
  echo "gpu=${GPU}"
  echo "history_tokens=${HISTORY_TOKENS}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "seed=${SEED}"
  echo "variant=${VARIANT}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
    "${ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py" \
    "${ROOT}/src/summarize_qksieve_resident_key_ab_20260807.py"
} >"${RUN_ROOT}/manifest.txt"

for mode in off on; do
  output="${RUN_ROOT}/${mode}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    continue
  fi
  mkdir -p "${output}/quality"
  if [[ "${mode}" == "on" ]]; then
    export QKSIEVE_BUILD_RESIDENT_KEY_FACTORS=1
  else
    unset QKSIEVE_BUILD_RESIDENT_KEY_FACTORS || true
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" \
  QKSIEVE_PROFILE_OUTPUT="${output}/index_profile.json" \
    "${PYTHON}" -u \
      "${ROOT}/src/profile_qksieve_realmodel_index_build_20260807.py" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_requestlocal_template.pt" \
      --output_dir "${output}/quality" \
      --history_tokens "${HISTORY_TOKENS}" \
      --stream_reference_history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --text_file "${TEXT_FILE}" \
      --repeat_topic_stream_if_short \
      --prefill_chunk_tokens 1024 \
      --protect_recent_tokens 0 \
      --dataset_cache_dir "${ROOT}/datasets" \
      --seed "${SEED}" \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANT}" \
      >"${RUN_ROOT}/logs/${mode}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
done

"${PYTHON}" "${ROOT}/src/summarize_qksieve_resident_key_ab_20260807.py" \
  --run_root "${RUN_ROOT}" >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
