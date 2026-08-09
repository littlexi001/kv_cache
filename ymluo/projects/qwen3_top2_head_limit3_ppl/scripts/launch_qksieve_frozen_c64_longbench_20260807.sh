#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/longbench_frozen_c64_smoke_20260807}"
PROMPT_WRAPPER="${PROMPT_WRAPPER:-qwen3}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-1}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-8}"
NUM_SHARDS="${NUM_SHARDS:-1}"
GPUS="${GPUS:-0}"
QK_WORKERS="${QK_WORKERS:-36}"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"

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
  echo "NUM_SHARDS=${NUM_SHARDS} needs at least that many GPUS" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model is incomplete: ${MODEL}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "LongBench manifest is missing: ${DATA_DIR}/manifest.json" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs"
rm -f "${RUN_ROOT}/FAILED"
{
  echo "root=${ROOT}"
  echo "model=${MODEL}"
  echo "data_dir=${DATA_DIR}"
  echo "prompt_wrapper=${PROMPT_WRAPPER}"
  echo "max_samples_per_task=${MAX_SAMPLES_PER_TASK}"
  echo "max_new_tokens_override=${MAX_NEW_TOKENS_OVERRIDE}"
  echo "num_shards=${NUM_SHARDS}"
  echo "gpus=${GPUS}"
  echo "qk_workers=${QK_WORKERS}"
  echo "methods=full_kv,${METHOD}"
  sha256sum \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${ROOT}/src/run_sample_calibrated_longbench_20260717.py" \
    "${ROOT}/src/summarize_qksieve_frozen_longbench_20260807.py" \
    "${DATA_DIR}/manifest.json"
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
    --methods "full_kv,${METHOD}" \
    --max_samples_per_task "${MAX_SAMPLES_PER_TASK}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_new_tokens_override "${MAX_NEW_TOKENS_OVERRIDE}" \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper "${PROMPT_WRAPPER}" \
    --minimum_sparse_prefix_tokens 0 \
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

expected_pairs="$("${PYTHON}" - "${DATA_DIR}/manifest.json" "${MAX_SAMPLES_PER_TASK}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
limit = int(sys.argv[2])
print(sum(min(count, limit) if limit > 0 else count for count in manifest["task_counts"].values()))
PY
)"
"${PYTHON}" "${ROOT}/src/summarize_qksieve_frozen_longbench_20260807.py" \
  --run_root "${RUN_ROOT}" \
  --expected_pairs "${expected_pairs}" \
  --expected_tasks 16 \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
