#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
TRACE_ROOT="${ROOT}/results/20260803_qksieve_qkv96k_2topic_gpu1234_v2/traces"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massladder_qwen96k_gpu23_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTPUT}/logs"
cd "${ROOT}"

run_case() {
  local gpu="$1"
  local topic="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/analyze_qksieve_output_risk_budget_20260803.py \
    --trace "${TRACE_ROOT}/${topic}.pt" \
    --output_dir "${OUTPUT}/${topic}" \
    --model_name_or_path "${MODEL}" \
    --device cuda \
    --fixed_top_k 1280 \
    --coverage_targets 0.90 \
    --mass_ladder_samples 1024 \
    --mass_ladder_growth 1.5 \
    --minimum_top_k 1280 \
    --maximum_top_k 0 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --risk_bits 4 \
    --focus_mass_ladder \
    >"${OUTPUT}/logs/${topic}.log" 2>&1
  touch "${OUTPUT}/${topic}_COMPLETE"
}

run_case 2 sports &
pid2=$!
run_case 3 medicine &
pid3=$!
wait "$pid2"
wait "$pid3"
touch "${OUTPUT}/ALL_COMPLETE"
