#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/frozen_python_fastpath_ab_32k_20260807}"
GPU="${GPU:-0}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-17}"
RUN_OLD="${RUN_OLD:-1}"
RUN_FAST="${RUN_FAST:-1}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_NATIVE_GQA_SDPA=decode
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_RESIDENT_VALUE_WORKERS=12
export QKSIEVE_PARALLEL_QK_WORKERS=36
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_CPROFILE_RANGE || true
unset QKSIEVE_CUDA_PROFILER_RANGE || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

run_condition() {
  local label="$1"
  local fastpath="$2"
  local output_dir="${RUN_ROOT}/${label}"
  mkdir -p "${output_dir}"
  QKSIEVE_FROZEN_DECODE_PYTHON_FASTPATH="${fastpath}" \
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
      "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_requestlocal_template.pt" \
      --output_dir "${output_dir}" \
      --history_tokens "${HISTORY_TOKENS}" \
      --stream_reference_history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --text_file "${TEXT_FILE}" \
      --repeat_topic_stream_if_short \
      --prefill_chunk_tokens 1024 \
      --protect_recent_tokens 0 \
      --dataset_cache_dir "${ROOT}/datasets" \
      --seed 20260851 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANT}" \
      >"${RUN_ROOT}/${label}.log" 2>&1
}

mkdir -p "${RUN_ROOT}"
if [[ "${RUN_OLD}" == "1" ]]; then
  run_condition old 0
fi
if [[ "${RUN_FAST}" == "1" ]]; then
  run_condition fast 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
