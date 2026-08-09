#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
EXAMPLES_JSONL="${EXAMPLES_JSONL:-${ROOT}/data/ruler_synthetic11_qwen4b_4k32k_m1.jsonl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/ruler_frozen_c64_synthetic11_m1_20260807}"
PROMPT_WRAPPER="${PROMPT_WRAPPER:-qwen3}"
TASKS="${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe}"
LENGTHS="${LENGTHS:-4096,8192,16384,32768}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-1}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPUS="${GPUS:-0,1,2,3}"
QK_WORKERS="${QK_WORKERS:-10}"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"

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
export QKSIEVE_PARALLEL_QK_WORKERS="${QK_WORKERS}"
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "NUM_SHARDS=${NUM_SHARDS} needs at least that many GPUs" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model is incomplete: ${MODEL}" >&2
  exit 1
fi
if [[ ! -s "${EXAMPLES_JSONL}" ]]; then
  echo "RULER examples are missing: ${EXAMPLES_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${EXAMPLES_JSONL}.manifest.json" ]]; then
  echo "RULER example manifest is missing: ${EXAMPLES_JSONL}.manifest.json" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED"
{
  echo "root=${ROOT}"
  echo "model=${MODEL}"
  echo "examples=${EXAMPLES_JSONL}"
  echo "prompt_wrapper=${PROMPT_WRAPPER}"
  echo "tasks=${TASKS}"
  echo "lengths=${LENGTHS}"
  echo "max_samples_per_task=${MAX_SAMPLES_PER_TASK}"
  echo "max_new_tokens_override=${MAX_NEW_TOKENS_OVERRIDE}"
  echo "num_shards=${NUM_SHARDS}"
  echo "gpus=${GPUS}"
  echo "qk_workers=${QK_WORKERS}"
  echo "methods=full_kv,${METHOD}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_sample_calibrated_longbench_20260717.py" \
    "${ROOT}/src/run_sample_calibrated_ruler_20260717.py" \
    "${ROOT}/src/summarize_qksieve_frozen_c64_ruler_20260807.py" \
    "${EXAMPLES_JSONL}" \
    "${EXAMPLES_JSONL}.manifest.json"
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
    "${ROOT}/src/run_sample_calibrated_ruler_20260717.py" \
    --model_name_or_path "${MODEL}" \
    --examples_jsonl "${EXAMPLES_JSONL}" \
    --output_dir "${output}" \
    --methods "full_kv,${METHOD}" \
    --ruler_tasks "${TASKS}" \
    --ruler_lengths "${LENGTHS}" \
    --max_samples_per_task "${MAX_SAMPLES_PER_TASK}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --max_new_tokens_override "${MAX_NEW_TOKENS_OVERRIDE}" \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper "${PROMPT_WRAPPER}" \
    --minimum_sparse_prefix_tokens 0 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
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

length_samples="$(${PYTHON} - "${LENGTHS}" "${MAX_SAMPLES_PER_TASK}" <<'PY'
import sys
print(",".join(f"{length}:{sys.argv[2]}" for length in sys.argv[1].split(",")))
PY
)"
"${PYTHON}" "${ROOT}/src/summarize_qksieve_frozen_c64_ruler_20260807.py" \
  --run_root "${RUN_ROOT}" \
  --expected_tasks "${TASKS}" \
  --expected_length_samples "${length_samples}" \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
