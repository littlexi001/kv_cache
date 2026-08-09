#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Meta-Llama-3.1-8B-Instruct-ms}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/LongBench}"
REFERENCE_RUN_ROOT="${REFERENCE_RUN_ROOT:-${ROOT}/results/longbench_frozen_c64_llama8b_m5_official_20260807}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/longbench_llama8b_weak3_tailalpha0_m5_20260807}"
GPUS="${GPUS:-0,1,2}"
NUM_SHARDS="${NUM_SHARDS:-3}"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="lcc,multifieldqa_en,qmsum"

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
export QKSIEVE_PARALLEL_QK_WORKERS=12
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.0
unset QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH || true
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "NUM_SHARDS=${NUM_SHARDS} needs at least that many GPUS" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED" "${RUN_ROOT}/ALL_COMPLETE"
{
  echo "purpose=strict tail-correction ablation with unchanged selector kernel"
  echo "model=${MODEL}"
  echo "tasks=${TASKS}"
  echo "methods=${METHOD}"
  echo "QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.0"
  echo "QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=unset"
  echo "reference_run_root=${REFERENCE_RUN_ROOT}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_critical_position_budget_probe_20260715.py" \
    "${ROOT}/src/run_sample_calibrated_longbench_20260717.py" \
    "${ROOT}/src/summarize_qksieve_valuesketch_weak_task_ab_20260807.py"
} >"${RUN_ROOT}/manifest.txt"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local output="${RUN_ROOT}/shard${shard}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    "${ROOT}/src/run_sample_calibrated_longbench_20260717.py" \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA_DIR}" \
    --output_dir "${output}" \
    --tasks "${TASKS}" \
    --methods "${METHOD}" \
    --max_samples_per_task 5 \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --minimum_sparse_prefix_tokens 0 \
    --collect_attention_stats \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    >"${RUN_ROOT}/logs/shard${shard}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  run_shard "${shard}" "${gpu_list[$shard]}" &
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
"${PYTHON}" \
  "${ROOT}/src/summarize_qksieve_valuesketch_weak_task_ab_20260807.py" \
  --reference_run_root "${REFERENCE_RUN_ROOT}" \
  --ab_run_root "${RUN_ROOT}" \
  --expected_pairs 15 \
  --ablation tail_alpha0 \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
