#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
FULL_ROOT="${ROOT}/results/20260715_full_reference_external_holdout"
RUNTIME_ROOT="${ROOT}/results/20260715_progressive_pca_mlp_external_holdout"
ROUTER="${ROOT}/results/20260715_progressive_pca_stability_mlp/router.joblib"
mkdir -p "${FULL_ROOT}" "${RUNTIME_ROOT}"

topics=(space politics)
for gpu in 0 1; do
  topic="${topics[$gpu]}"
  (
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${ROOT}/src" \
      PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
      "${PYTHON_BIN}" "${ROOT}/src/run_critical_position_budget_probe_20260715.py" \
        --model_name_or_path "${MODEL}" \
        --output_dir "${FULL_ROOT}/${topic}_w0" \
        --topics "${topic}" \
        --window_indices 0 \
        --only_full

    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${ROOT}/src" \
      PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
      "${PYTHON_BIN}" "${ROOT}/src/run_progressive_pca_router_ppl_20260715.py" \
        --model_name_or_path "${MODEL}" \
        --router_path "${ROUTER}" \
        --full_root "${FULL_ROOT}" \
        --output_dir "${RUNTIME_ROOT}/${topic}_w0" \
        --topics "${topic}" \
        --window_indices 0 \
        --projection_dim 32
  ) >"${RUNTIME_ROOT}/${topic}_w0_queue.log" 2>&1 </dev/null &
  echo "launched queue pid=$! gpu=${gpu} topic=${topic}"
done
