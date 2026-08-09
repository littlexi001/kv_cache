#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/native_gqa_full_ab_32k_20260807}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
LEGACY_GPU="${LEGACY_GPU:-0}"
NATIVE_GPU="${NATIVE_GPU:-1}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
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

mkdir -p "${RUN_ROOT}/logs"
{
  echo "purpose=Full attention repeat_kv versus native GQA A/B"
  echo "history_tokens=${HISTORY_TOKENS}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "legacy_gpu=${LEGACY_GPU}"
  echo "native_gpu=${NATIVE_GPU}"
  echo "text_file=${TEXT_FILE}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py"
} >"${RUN_ROOT}/manifest.txt"

run_mode() {
  local mode="$1"
  local native_gqa="$2"
  local gpu="$3"
  local output="${RUN_ROOT}/${mode}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  QKSIEVE_NATIVE_GQA_SDPA="${native_gqa}" \
    "${PYTHON}" -u \
      "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_requestlocal_template.pt" \
      --output_dir "${output}" \
      --history_tokens "${HISTORY_TOKENS}" \
      --stream_reference_history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --text_file "${TEXT_FILE}" \
      --repeat_topic_stream_if_short \
      --prefill_chunk_tokens 1024 \
      --dataset_cache_dir "${ROOT}/datasets" \
      --seed 20260807 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANT}" \
      --full_only \
      >"${RUN_ROOT}/logs/${mode}.log" 2>&1
}

run_mode legacy off "${LEGACY_GPU}" &
legacy_pid=$!
run_mode native_gqa decode "${NATIVE_GPU}" &
native_pid=$!

wait "${legacy_pid}"
wait "${native_pid}"
touch "${RUN_ROOT}/ALL_COMPLETE"
