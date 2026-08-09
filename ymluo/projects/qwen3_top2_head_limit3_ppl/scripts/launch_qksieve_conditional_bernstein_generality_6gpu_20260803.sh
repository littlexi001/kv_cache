#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260803_conditional_bernstein_generality_6gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/home/fdong/miniconda3/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local name="$2"
  local trace="$3"
  local model="$4"
  local top_k="$5"
  local records="$6"
  local output="${RUN_ROOT}/${name}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_tail_partition_calibration_20260803.py \
    --traces "${trace}" \
    --output_dir "${output}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --top_k "${top_k}" \
    --sample_counts 256 \
    --block_sizes 131072 \
    --conditional_dims 8,16 \
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
    --max_records_per_trace "${records}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1
  touch "${output}/ALL_COMPLETE"
}

run_case 0 llama_religion4k \
  "${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt" \
  "${LLAMA_MODEL}" 256 8 & p0=$!
run_case 1 qwen_sports32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/sports.pt" \
  "${QWEN_MODEL}" 1280 8 & p1=$!
run_case 2 qwen_medicine32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/medicine.pt" \
  "${QWEN_MODEL}" 1280 8 & p2=$!
run_case 3 qwen_sports96k \
  "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/sports.pt" \
  "${QWEN_MODEL}" 1280 8 & p3=$!
run_case 4 qwen_medicine96k \
  "${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces/medicine.pt" \
  "${QWEN_MODEL}" 1280 8 & p4=$!
run_case 5 llama_computer128k \
  "${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt" \
  "${LLAMA_MODEL}" 1280 1 & p5=$!

failed=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${RUN_ROOT}/FAILED"
  exit 1
fi
touch "${RUN_ROOT}/ALL_COMPLETE"
