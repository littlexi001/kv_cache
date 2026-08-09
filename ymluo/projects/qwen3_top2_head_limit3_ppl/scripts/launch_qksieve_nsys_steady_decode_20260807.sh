#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/nsys_steady_decode_32k_20260807}"
GPU="${GPU:-0}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-8}"
SEED="${SEED:-20260851}"
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
export QKSIEVE_PARALLEL_QK_WORKERS=36
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_CUDA_PROFILER_RANGE=1
export QKSIEVE_CUDA_PROFILER_METHOD="${QKSIEVE_CUDA_PROFILER_METHOD:-direct_countcap}"
export QKSIEVE_CUDA_PROFILER_START_STEP="${QKSIEVE_CUDA_PROFILER_START_STEP:-2}"
export QKSIEVE_CUDA_PROFILER_STEPS="${QKSIEVE_CUDA_PROFILER_STEPS:-4}"
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_PROFILE_INDEX_HASHES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true
unset QKSIEVE_BUILD_RESIDENT_KEY_FACTORS || true
unset QKSIEVE_RESIDENT_KEY_MOMENTS_ONLY || true

mkdir -p "${RUN_ROOT}/quality" "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/ALL_COMPLETE"
{
  echo "purpose=Nsight Systems frozen-QKSieve steady decode kernel profile"
  echo "history_tokens=${HISTORY_TOKENS}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "profile_start_step=${QKSIEVE_CUDA_PROFILER_START_STEP}"
  echo "profile_steps=${QKSIEVE_CUDA_PROFILER_STEPS}"
  echo "profile_method=${QKSIEVE_CUDA_PROFILER_METHOD}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py"
} >"${RUN_ROOT}/manifest.txt"

CUDA_VISIBLE_DEVICES="${GPU}" nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="${RUN_ROOT}/qksieve_steady" \
  "${PYTHON}" -u \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
    --model_name_or_path "${MODEL}" \
    --template "${ROOT}/nonexistent_requestlocal_template.pt" \
    --output_dir "${RUN_ROOT}/quality" \
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
    >"${RUN_ROOT}/logs/profile.log" 2>&1

nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output "${RUN_ROOT}/cuda_gpu_kern_sum" \
  "${RUN_ROOT}/qksieve_steady.nsys-rep" \
  >"${RUN_ROOT}/logs/stats.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
