#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_TEMPLATE="${QWEN_TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
LLAMA_TEMPLATE="${LLAMA_TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_sampledfused_triprobe_3gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

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
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local template="$4"
  local topic="$5"
  local history="$6"
  local budget="$7"
  local seed="$8"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${model}" \
    --template "${template}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --history_tokens "${history}" \
    --stream_reference_history_tokens "${history}" \
    --eval_tokens "${EVAL_TOKENS:-4}" \
    --topic "${topic}" \
    --seed "${seed}" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_memory_per_gpu_gib 22 \
    --variants "qksieve_qmse_oas_requestlocal_valuesketch16_k${budget},qksieve_qmse_oas_requestlocal_valuesketch16_sampled_k${budget}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
  touch "${RUN_ROOT}/${name}_COMPLETE"
}

run_case 5 qwen_medicine32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" medicine 32768 1280 20261403 & p0=$!
run_case 6 qwen_sports32k "${QWEN_MODEL}" "${QWEN_TEMPLATE}" sports 32768 1280 20261404 & p1=$!
run_case 7 llama_religion4k "${LLAMA_MODEL}" "${LLAMA_TEMPLATE}" religion 4096 256 20261405 & p2=$!

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
