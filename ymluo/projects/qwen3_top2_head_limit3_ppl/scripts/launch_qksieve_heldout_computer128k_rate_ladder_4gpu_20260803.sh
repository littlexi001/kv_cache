#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE="${TRACE:-${ROOT}/results/20260801_real_qkv_trace_computer128k/computer.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_heldout_computer128k_rate_ladder_4gpu_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_rate() {
  local gpu="$1"
  local rate="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${TRACE}" \
    --output_dir "${OUTPUT}/computer128k/rate${rate}" \
    --model_name_or_path "${MODEL}" \
    --device cuda \
    --fixed_top_ks 1280 --coverage_targets 0.9 \
    --minimum_top_k 1 \
    --key_rate_budget "${rate}" --key_quantizer plain \
    --key_allocation_objective oas_qk_mse \
    --key_allocation_query_source basis \
    --query_factor_source prefill --query_factor_prefill_tokens 8 \
    --balanced_rss_tolerances 0.0025 --rss_safety_factors 2 \
    --focus_balanced_rss \
    >"${OUTPUT}/logs/rate${rate}.log" 2>&1
  touch "${OUTPUT}/rate${rate}_COMPLETE"
}

run_rate 0 15 & pid0=$!
run_rate 1 19 & pid1=$!
run_rate 2 23 & pid2=$!
run_rate 3 27 & pid3=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then touch "${OUTPUT}/FAILED"; exit 1; fi
"${PYTHON}" src/summarize_qksieve_output_probe_rate_controller_20260803.py \
  --input_root "${OUTPUT}" \
  --output_dir "${OUTPUT}/controller"
touch "${OUTPUT}/ALL_COMPLETE"
