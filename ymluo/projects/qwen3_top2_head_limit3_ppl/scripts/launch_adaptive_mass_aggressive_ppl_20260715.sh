#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_adaptive_mass_aggressive_32k_w3}"

mkdir -p "${RUN_ROOT}"

run_topic() {
  local topic="$1"
  local gpu="$2"
  for window in 0 1 2; do
    name="${topic}_w${window}"
    output_dir="${RUN_ROOT}/${name}"
    log_path="${RUN_ROOT}/${name}.log"
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${ROOT}/src" \
      "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output_dir}" \
        --topics "${topic}" \
        --window_indices "${window}" \
        --history_tokens 32000 \
        --query_tokens 256 \
        --eval_tokens 256 \
        --window_stride_tokens 32512 \
        --mass_thresholds 0.75,0.80,0.85,0.875 \
        --budget_fractions 0.0025,0.005,0.01,0.02,0.04 \
        --prefill_chunk_tokens 2048 \
        >"${log_path}" 2>&1
  done
}

run_topic sports 0 &
sports_pid=$!
run_topic medicine 7 &
medicine_pid=$!
echo "launched sports queue pid=${sports_pid} gpu=0"
echo "launched medicine queue pid=${medicine_pid} gpu=7"
wait
