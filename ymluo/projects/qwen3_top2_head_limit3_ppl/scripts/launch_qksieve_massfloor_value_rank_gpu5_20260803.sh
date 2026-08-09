#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massfloor_value_rank_gpu5_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.6

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local name="$1"
  local trace="$2"
  local model="$3"
  local rank="$4"
  local case_output="${OUTPUT}/${name}_r${rank}"
  mkdir -p "${case_output}"
  CUDA_VISIBLE_DEVICES=5 "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${trace}" \
    --output_dir "${case_output}" \
    --model_name_or_path "${model}" \
    --device cuda \
    --fixed_top_k 1280 \
    --fixed_top_ks 1280,2560 \
    --coverage_targets 0.90,0.95 \
    --minimum_top_k 1280 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank "${rank}" \
    --value_bits 4 \
    --risk_bits 4 \
    --score_calibration_samples 256 \
    >"${OUTPUT}/logs/${name}_r${rank}.log" 2>&1
  touch "${OUTPUT}/${name}_r${rank}_COMPLETE"
}

for rank in 32 64; do
  run_case \
    computer128k \
    "${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt" \
    "${LLAMA_MODEL}" \
    "${rank}"
  run_case \
    sports32k \
    "${ROOT}/results/20260803_prefill_tail_calibration_qwen32k_2topic_v1/traces/sports.pt" \
    "${QWEN_MODEL}" \
    "${rank}"
done

touch "${OUTPUT}/ALL_COMPLETE"
