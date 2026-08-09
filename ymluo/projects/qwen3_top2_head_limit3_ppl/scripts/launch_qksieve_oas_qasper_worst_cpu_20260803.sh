#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE="${TRACE:-${ROOT}/results/20260801_qksieve_qasper_reference_trace_m10_gpu4/traces/qasper__9fb085a1f47673d1907f2378c90843b4b6e8622a14fe1fa9__qksieve_fullprompt_auto_plain_fulltopk.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_qasper_worst_oas_cpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-24}"

mkdir -p "${OUTPUT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="" "${PYTHON}" -u \
  src/analyze_qksieve_output_risk_budget_20260803.py \
  --trace "${TRACE}" \
  --output_dir "${OUTPUT}/plain_oas_qk_mse_rate19" \
  --model_name_or_path "${MODEL}" \
  --device cpu \
  --fixed_top_k 1280 \
  --coverage_targets 0.95 \
  --coverage_histogram_bins 256 \
  --minimum_top_k 256 \
  --maximum_top_k 0 \
  --key_rate_budget 19 \
  --key_quantizer plain \
  --key_allocation_objective oas_qk_mse \
  --value_rank 16 \
  --value_bits 4 \
  --risk_bits 4 \
  --query_factor_source prefill \
  --query_factor_prefill_tokens 8 \
  --focus_mass_ladder \
  >"${OUTPUT}/run.log" 2>&1

touch "${OUTPUT}/ALL_COMPLETE"
