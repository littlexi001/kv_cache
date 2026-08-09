#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
ALPHA1_ROOT="${ALPHA1_ROOT:-${ROOT}/results/frozen_stage_profile_32k128k_20260807}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/tailalpha0_length_32k128k_20260807}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"
LENGTHS=(32768 65536 131072)
GPU_GROUPS=(2 7 3,4)

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
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.0
unset QKSIEVE_BUILD_RESIDENT_KEY_FACTORS || true
unset QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH || true
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED" "${RUN_ROOT}/ALL_COMPLETE"
{
  echo "purpose=quality-only tail alpha length ablation"
  echo "alpha1_root=${ALPHA1_ROOT}"
  echo "lengths=${LENGTHS[*]}"
  echo "gpu_groups=${GPU_GROUPS[*]}"
  echo "QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.0"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
    "${ROOT}/src/summarize_qksieve_tailalpha_length_ab_20260807.py"
} >"${RUN_ROOT}/manifest.txt"

run_length() {
  local history="$1"
  local gpu_group="$2"
  local output="${RUN_ROOT}/n${history}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu_group}" "${PYTHON}" -u \
    "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
    --model_name_or_path "${MODEL}" \
    --template "${ROOT}/nonexistent_requestlocal_template.pt" \
    --output_dir "${output}" \
    --history_tokens "${history}" \
    --stream_reference_history_tokens "${history}" \
    --eval_tokens 16 \
    --text_file "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${ROOT}/datasets" \
    --seed 20260843 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANT}" \
    >"${RUN_ROOT}/logs/n${history}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

pids=()
for index in "${!LENGTHS[@]}"; do
  run_length "${LENGTHS[$index]}" "${GPU_GROUPS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
"${PYTHON}" "${ROOT}/src/summarize_qksieve_tailalpha_length_ab_20260807.py" \
  --alpha1_root "${ALPHA1_ROOT}" \
  --alpha0_root "${RUN_ROOT}" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
