#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TRACE="${TRACE:-${ROOT}/results/20260803_llama4k_religion_all32_qkv_trace_v1/traces/religion.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_keyrate_religion4k_3gpu_v1}"
GPU0="${GPU0:-3}"
GPU1="${GPU1:-4}"
GPU2="${GPU2:-5}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_rate() {
  local gpu="$1"
  local rate="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${TRACE}" \
    --output_dir "${OUTPUT}/rate${rate}" \
    --model_name_or_path "${MODEL}" \
    --device cuda \
    --fixed_top_k 1280 \
    --coverage_targets 0.95,0.975 \
    --coverage_histogram_bins 256 \
    --minimum_top_k 256 \
    --maximum_top_k 0 \
    --key_rate_budget "${rate}" \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    --focus_mass_ladder \
    >"${OUTPUT}/logs/rate${rate}.log" 2>&1
  touch "${OUTPUT}/rate${rate}_COMPLETE"
}

run_rate "${GPU0}" 15 & pid0=$!
run_rate "${GPU1}" 19 & pid1=$!
run_rate "${GPU2}" 23 & pid2=$!

failed=0
for pid in "${pid0}" "${pid1}" "${pid2}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  touch "${OUTPUT}/FAILED"
  exit 1
fi
touch "${OUTPUT}/ALL_COMPLETE"
