#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Meta-Llama-3.1-8B-Instruct-ms}"
DATA_DIR="${DATA_DIR:-${ROOT}/experiments/frozen_c64_20260807/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260810_qksieve_robust_llama_full_longbench_v1}"
RUNNER="${ROOT}/src/run_sample_calibrated_longbench_20260717.py"
SUMMARIZER="${ROOT}/src/summarize_qksieve_robust_longbench_20260810.py"
FROZEN_CONFIG="${ROOT}/configs/qksieve_robust_iclr2027_frozen_20260810.json"
METHOD="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64"
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

export PATH="$(dirname "${PYTHON}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export QKSIEVE_TRUST_REMOTE_CODE=0
export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_PARALLEL_QK_WORKERS="${QKSIEVE_PARALLEL_QK_WORKERS:-8}"
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION=1
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT=512
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=0.5
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=0
unset QKSIEVE_PROFILE_STAGES QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
touch "${RUN_ROOT}/RUNNING"
rm -f "${RUN_ROOT}/ALL_COMPLETE" "${RUN_ROOT}/FAILED"

fail() {
  touch "${RUN_ROOT}/FAILED"
  rm -f "${RUN_ROOT}/RUNNING"
  exit 1
}

if [[ ! -f "${MODEL}/config.json" ]] || \
   [[ ! -f "${DATA_DIR}/manifest.json" ]] || \
   [[ ! -f "${FROZEN_CONFIG}" ]]; then
  echo "model, LongBench data, or frozen config is incomplete" \
    >"${RUN_ROOT}/logs/input_error.log"
  fail
fi

expected_pairs="$("${PYTHON}" - "${DATA_DIR}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(int(value) for value in manifest["task_counts"].values()))
PY
)" || fail
if [[ "${expected_pairs}" -ne 3750 ]]; then
  echo "expected 3750 LongBench rows, found ${expected_pairs}" \
    >"${RUN_ROOT}/logs/input_error.log"
  fail
fi

{
  echo "schema=qksieve_robust_llama_full_longbench_v1"
  echo "source_tree_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  "${PYTHON}" - "${FROZEN_CONFIG}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"numerical_freeze_commit_sha={payload['numerical_freeze_commit_sha']}")
print(
    "audited_implementation_commit_sha="
    f"{payload['audited_implementation_commit_sha']}"
)
PY
  echo "model=${MODEL}"
  echo "prompt_wrapper=llama3"
  echo "prompt_truncation=official_middle_7500"
  echo "expected_pairs=${expected_pairs}"
  echo "method=${METHOD}"
  echo "max_quantile_samples=${QKSIEVE_MAX_QUANTILE_SAMPLE_COUNT}"
  echo "value_tail_alpha=${QKSIEVE_VALUE_SKETCH_TAIL_ALPHA}"
  sha256sum \
    "${MODEL}/config.json" \
    "${DATA_DIR}/manifest.json" \
    "${FROZEN_CONFIG}" \
    "${ROOT}/src/run_head_top2_targeted_ppl_20260714.py" \
    "${RUNNER}" "${SUMMARIZER}"
} >"${RUN_ROOT}/manifest.txt" || fail

IFS=',' read -r -a gpu_list <<<"${GPUS}"
if [[ "${#gpu_list[@]}" -ne 8 ]]; then
  echo "formal protocol requires exactly eight GPUs" \
    >"${RUN_ROOT}/logs/hardware_error.log"
  fail
fi

run_shard() {
  local shard="$1" gpu="$2" output="${RUN_ROOT}/shard${1}"
  if [[ -f "${output}/ALL_COMPLETE" ]]; then return; fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${RUNNER}" \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA_DIR}" \
    --output_dir "${output}" \
    --tasks "${TASKS}" \
    --methods "full_kv,${METHOD}" \
    --max_samples_per_task 0 \
    --sample_offset_per_task 0 \
    --num_shards 8 --shard_index "${shard}" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --minimum_sparse_prefix_tokens 0 \
    --collect_attention_stats \
    --dtype float16 --device cuda --device_map auto \
    --max_memory_per_gpu_gib 22 \
    >"${RUN_ROOT}/logs/shard${shard}.log" 2>&1 || return 1
  touch "${output}/ALL_COMPLETE"
}

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  run_shard "${shard}" "${gpu_list[$shard]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" -eq 0 ]] || fail

"${PYTHON}" "${SUMMARIZER}" \
  --run_root "${RUN_ROOT}" \
  --expected_pairs 3750 \
  --expected_tasks 16 \
  --bootstrap_resamples 10000 \
  --seed 20260810 \
  >"${RUN_ROOT}/logs/summarize.log" 2>&1 || fail

rm -f "${RUN_ROOT}/RUNNING" "${RUN_ROOT}/FAILED"
touch "${RUN_ROOT}/ALL_COMPLETE"
