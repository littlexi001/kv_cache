#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tiled_workspace_ab_32k_20260807}"
GPU="${GPU:-0}"
HISTORY="${HISTORY:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
REPEATS="${REPEATS:-3}"
SEED="${SEED:-20260847}"
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
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
{
  echo "purpose=quality-equivalent resident-strided and warp-tiled ValueSketch A/B"
  echo "history=${HISTORY}"
  echo "eval_tokens=${EVAL_TOKENS}"
  echo "repeats=${REPEATS}"
  echo "seed=${SEED}"
  echo "variant=${VARIANT}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/qksieve_valuesketch_cuda_20260801.py" \
    "${ROOT}/src/run_direct_countcap_denseprompt_ppl_20260725.py" \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py"
} >"${RUN_ROOT}/manifest.txt"

run_case() {
  local repeat="$1"
  local mode="$2"
  local resident="$3"
  local tiled="$4"
  local output="${RUN_ROOT}/r${repeat}/${mode}"
  local log="${RUN_ROOT}/logs/r${repeat}_${mode}.log"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE="${resident}" \
  QKSIEVE_TILED_VALUE_ATTENTION="${tiled}" \
    "${PYTHON}" -u \
      "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_requestlocal_template.pt" \
      --output_dir "${output}" \
      --history_tokens "${HISTORY}" \
      --stream_reference_history_tokens "${HISTORY}" \
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
      >"${log}" 2>&1
  touch "${output}/ALL_COMPLETE"
}

for ((repeat=1; repeat<=REPEATS; repeat++)); do
  if (( repeat % 2 )); then
    order=(legacy_allocating resident_strided_scalar resident_strided_tiled)
  else
    order=(resident_strided_tiled resident_strided_scalar legacy_allocating)
  fi
  for mode in "${order[@]}"; do
    case "${mode}" in
      legacy_allocating) run_case "${repeat}" "${mode}" 0 0 ;;
      resident_strided_scalar) run_case "${repeat}" "${mode}" 1 0 ;;
      resident_strided_tiled) run_case "${repeat}" "${mode}" 1 1 ;;
      *) echo "Unknown mode: ${mode}" >&2; exit 1 ;;
    esac
  done
done

"${PYTHON}" "${ROOT}/src/summarize_qksieve_tiled_workspace_ab_20260807.py" \
  "${RUN_ROOT}"
touch "${RUN_ROOT}/ALL_COMPLETE"
