#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_v40_c64_final_profile_5gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANT="qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpus="$1"
  local name="$2"
  local length="$3"
  local eval_tokens="$4"
  local profile="$5"
  local device_map="$6"
  local output_dir="${RUN_ROOT}/${name}"
  if [[ "${profile}" == "1" ]]; then
    export QKSIEVE_PROFILE_STAGES=1
  else
    unset QKSIEVE_PROFILE_STAGES || true
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens "${length}" \
    --stream_reference_history_tokens "${length}" \
    --eval_tokens "${eval_tokens}" \
    --topic mixed_b \
    --seed "$((20262000 + length / 1024))" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map "${device_map}" \
    --max_memory_per_gpu_gib 22 \
    --variants "${VARIANT}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

run_case 0 n32768_eval32 32768 32 0 auto & p0=$!
run_case 1,2 n65536_profile8 65536 8 1 balanced & p1=$!
run_case 3,4 n131064_profile8 131064 8 1 balanced & p2=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
