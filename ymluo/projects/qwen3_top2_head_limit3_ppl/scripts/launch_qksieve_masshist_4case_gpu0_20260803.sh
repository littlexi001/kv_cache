#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_masshist_4case_gpu0_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local name="$1"
  local trace="$2"
  local model="$3"
  local case_output="${OUTPUT}/${name}"
  mkdir -p "${case_output}"
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${trace}" \
    --output_dir "${case_output}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --fixed_top_k 1280 \
    --fixed_top_ks 1280,2560 \
    --coverage_targets 0.90,0.95 \
    --coverage_histogram_bins 32,64,128,256 \
    --minimum_top_k 1280 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    --score_calibration_samples 256 \
    >"${OUTPUT}/logs/${name}.log" 2>&1
  touch "${OUTPUT}/${name}_COMPLETE"
}

run_case \
  religion4k \
  "${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt" \
  "${LLAMA_MODEL}"
run_case \
  sports32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/sports.pt" \
  "${QWEN_MODEL}"
run_case \
  medicine32k \
  "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/medicine.pt" \
  "${QWEN_MODEL}"
run_case \
  computer128k \
  "${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt" \
  "${LLAMA_MODEL}"

touch "${OUTPUT}/ALL_COMPLETE"
