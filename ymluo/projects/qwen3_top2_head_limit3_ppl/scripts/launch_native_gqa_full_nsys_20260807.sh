#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_pre_indexopt_20260807.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/nsys_native_gqa_full_32k_20260807}"
GPU="${GPU:-0}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-8}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_NATIVE_GQA_SDPA=decode
export QKSIEVE_CUDA_PROFILER_RANGE=1
export QKSIEVE_CUDA_PROFILER_METHOD=full_attention
export QKSIEVE_CUDA_PROFILER_START_STEP="${QKSIEVE_CUDA_PROFILER_START_STEP:-2}"
export QKSIEVE_CUDA_PROFILER_STEPS="${QKSIEVE_CUDA_PROFILER_STEPS:-4}"

mkdir -p "${RUN_ROOT}/quality" "${RUN_ROOT}/logs"
{
  echo "purpose=Nsight Systems native-GQA Full steady decode profile"
  echo "history_tokens=${HISTORY_TOKENS}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "native_gqa_mode=${QKSIEVE_NATIVE_GQA_SDPA}"
  echo "profile_start_step=${QKSIEVE_CUDA_PROFILER_START_STEP}"
  echo "profile_steps=${QKSIEVE_CUDA_PROFILER_STEPS}"
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
  --output="${RUN_ROOT}/native_gqa_full_steady" \
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
    --dataset_cache_dir "${ROOT}/datasets" \
    --seed 20260807 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --full_only \
    >"${RUN_ROOT}/logs/profile.log" 2>&1

nsys stats \
  --report cuda_gpu_kern_sum \
  --format csv \
  --output "${RUN_ROOT}/cuda_gpu_kern_sum" \
  "${RUN_ROOT}/native_gqa_full_steady.nsys-rep" \
  >"${RUN_ROOT}/logs/kernel_stats.log" 2>&1
nsys stats \
  --report cuda_api_sum \
  --format csv \
  --output "${RUN_ROOT}/cuda_api_sum" \
  "${RUN_ROOT}/native_gqa_full_steady.nsys-rep" \
  >"${RUN_ROOT}/logs/api_stats.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
