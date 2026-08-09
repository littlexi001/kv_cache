#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_progressive_pca_router_runtime_w2}"
ROUTER="${ROOT}/results/20260715_progressive_pca_router_strict/router.joblib"
FULL_ROOT="${ROOT}/results/20260715_critical_position_budget_32k_w3"
mkdir -p "${RUN_ROOT}"

topics=(sports medicine)
for gpu in 0 1; do
  topic="${topics[$gpu]}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_progressive_pca_router_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --router_path "${ROUTER}" \
      --full_root "${FULL_ROOT}" \
      --output_dir "${RUN_ROOT}/${topic}_w2" \
      --topics "${topic}" \
      --window_indices 2 \
      --projection_dim 32 \
      >"${RUN_ROOT}/${topic}_w2.log" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} topic=${topic}"
done
