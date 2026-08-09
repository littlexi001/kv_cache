#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_query_crossfit_multistep_8gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/home/fdong/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/traces"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local model="$3"
  local topic="$4"
  local history="$5"
  local layers="$6"
  local top_k="$7"
  local trace="${RUN_ROOT}/traces/${name}.pt"
  local output="${RUN_ROOT}/${name}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "${model}" \
    --output_path "${trace}" \
    --topic "${topic}" \
    --history_tokens "${history}" \
    --steps 8 \
    --layers "${layers}" \
    --prefill_query_tail_tokens 8 \
    --prefill_chunk_tokens 1024 \
    --seed 20260841 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"${RUN_ROOT}/logs/${name}_capture.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_tail_partition_calibration_20260803.py \
    --traces "${trace}" \
    --output_dir "${output}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --top_k "${top_k}" \
    --sample_counts 256 \
    --block_sizes 131072 \
    --conditional_dims 8 \
    --conditional_fit_stride 32 \
    --tail_sampling random \
    --key_sample_stride 32 \
    --value_sample_stride 32 \
    --query_shrinkage 0.5 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --value_metric wo_group \
    --risk_delta 0.01 \
    >"${RUN_ROOT}/logs/${name}_analyze.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

qwen_layers="0,8,17,26,35"
llama_layers="0,8,16,24,31"
run_case 0 qwen_sports32k "${QWEN_MODEL}" sports 32000 "${qwen_layers}" 1280 & p0=$!
run_case 1 qwen_medicine32k "${QWEN_MODEL}" medicine 32000 "${qwen_layers}" 1280 & p1=$!
run_case 2 qwen_sports96k "${QWEN_MODEL}" sports 96000 "${qwen_layers}" 1280 & p2=$!
run_case 3 qwen_medicine96k "${QWEN_MODEL}" medicine 96000 "${qwen_layers}" 1280 & p3=$!
run_case 4 llama_religion4k "${LLAMA_MODEL}" religion 3968 "${llama_layers}" 256 & p4=$!
run_case 5 llama_computer32k "${LLAMA_MODEL}" computer 32000 "${llama_layers}" 1280 & p5=$!
run_case 6 qwen_religion32k "${QWEN_MODEL}" religion 32000 "${qwen_layers}" 1280 & p6=$!
run_case 7 qwen_computer32k "${QWEN_MODEL}" computer 32000 "${qwen_layers}" 1280 & p7=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}" "${p6}" "${p7}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
