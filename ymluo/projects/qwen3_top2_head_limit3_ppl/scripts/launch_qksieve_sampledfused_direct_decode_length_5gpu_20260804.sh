#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_sampledfused_direct_decode_length_5gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"

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
  local gpus="$1"
  local length="$2"
  local device_map="$3"
  local output_dir="${RUN_ROOT}/n${length}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON}" -u \
    src/run_qksieve_coldskip_longcontext_quality_20260730.py \
    --model_name_or_path "${MODEL}" \
    --template "${TEMPLATE}" \
    --output_dir "${output_dir}" \
    --history_tokens "${length}" \
    --stream_reference_history_tokens "${length}" \
    --eval_tokens "${EVAL_TOKENS}" \
    --topic mixed_b \
    --seed "$((20261800 + length / 1024))" \
    --repeat_topic_stream_if_short \
    --prefill_chunk_tokens 1024 \
    --protect_recent_tokens 0 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    --dtype float16 \
    --device cuda \
    --device_map "${device_map}" \
    --max_memory_per_gpu_gib 22 \
    --variants qksieve_qmse_oas_requestlocal_valuesketch16_sampled_k1280 \
    >"${RUN_ROOT}/logs/n${length}.log" 2>&1
  touch "${output_dir}/ALL_COMPLETE"
}

run_case 0 32768 auto & p0=$!
run_case 1,2 65536 balanced & p1=$!
run_case 3,4 131040 balanced & p2=$!

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
