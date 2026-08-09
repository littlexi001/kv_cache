#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-${ROOT}/models/Qwen3-4B-Instruct-2507}"
TEXT_FILE="${TEXT_FILE:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/index_profile_32k_20260807}"
GPU="${GPU:-0}"
SOLVERS="${SOLVERS:-legacy legacy_cuda cholesky_cuda}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32768}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"
DRIVER="${DRIVER:-src/profile_qksieve_realmodel_index_build_20260807.py}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
if [[ "${QKSIEVE_PROFILE_STAGES:-0}" != "1" ]]; then
  unset QKSIEVE_PROFILE_STAGES || true
fi

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_solver() {
  local solver="$1"
  local output_dir="${RUN_ROOT}/${solver}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    QKSIEVE_QK_FACTOR_SOLVER="${solver}" \
    QKSIEVE_PROFILE_OUTPUT="${output_dir}/index_profile.json" \
    "${PYTHON}" -u \
      "${DRIVER}" \
      --model_name_or_path "${MODEL}" \
      --template "${ROOT}/nonexistent_template.pt" \
      --output_dir "${output_dir}/quality" \
      --history_tokens "${HISTORY_TOKENS}" \
      --stream_reference_history_tokens "${HISTORY_TOKENS}" \
      --eval_tokens "${EVAL_TOKENS}" \
      --text_file "${TEXT_FILE}" \
      --repeat_topic_stream_if_short \
      --prefill_chunk_tokens 1024 \
      --protect_recent_tokens 0 \
      --dataset_cache_dir "${ROOT}/datasets" \
      --seed 20260807 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      --max_memory_per_gpu_gib 22 \
      --variants "${VARIANT}" \
      >"${RUN_ROOT}/logs/${solver}.log" 2>&1
}

for solver in ${SOLVERS}; do
  run_solver "${solver}"
done

touch "${RUN_ROOT}/ALL_COMPLETE"
