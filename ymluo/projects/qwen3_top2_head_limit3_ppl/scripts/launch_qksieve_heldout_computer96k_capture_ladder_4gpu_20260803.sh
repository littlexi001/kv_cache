#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_heldout_computer96k_capture_ladder_4gpu_v1}"
TRACE="${OUTPUT}/traces/computer96k.pt"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/traces"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u \
  src/collect_real_qk_trace_20260715.py \
  --model_name_or_path "${MODEL}" \
  --output_path "${TRACE}" \
  --topic computer --history_tokens 96000 --steps 1 \
  --layers 0,7,14,21,28,35 \
  --prefill_query_tail_tokens 8 --prefill_chunk_tokens 256 \
  --seed 20260843 --dtype float16 --device cuda --device_map auto \
  >"${OUTPUT}/logs/capture.log" 2>&1
touch "${OUTPUT}/CAPTURE_COMPLETE"

run_rate() {
  local gpu="$1"
  local rate="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${TRACE}" \
    --output_dir "${OUTPUT}/computer96k/rate${rate}" \
    --model_name_or_path "${MODEL}" --device cuda \
    --fixed_top_ks 1280 --coverage_targets 0.9 --minimum_top_k 1 \
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
  --input_root "${OUTPUT}" --output_dir "${OUTPUT}/controller"
touch "${OUTPUT}/ALL_COMPLETE"
