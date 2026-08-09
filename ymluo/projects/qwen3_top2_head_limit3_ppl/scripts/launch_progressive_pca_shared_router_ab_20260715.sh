#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
FULL_ROOT="${ROOT}/results/20260715_critical_position_budget_32k_w3"
RUN_ROOT="${ROOT}/results/20260715_progressive_pca_shared_router_ab_w2"
mkdir -p "${RUN_ROOT}"

topics=(sports medicine sports medicine)
routers=(
  "${ROOT}/results/20260715_progressive_pca_stability_mlp/router.joblib"
  "${ROOT}/results/20260715_progressive_pca_stability_mlp/router.joblib"
  "${ROOT}/results/20260715_progressive_pca_shared_mean_relative_router/router.joblib"
  "${ROOT}/results/20260715_progressive_pca_shared_mean_relative_router/router.joblib"
)
labels=(old_mlp old_mlp relative_mlp relative_mlp)
gpus=(${GPUS:-1 2 3 4})
for slot in ${SLOTS:-0 1 2 3}; do
  gpu="${gpus[$slot]}"
  topic="${topics[$slot]}"
  label="${labels[$slot]}"
  name="${label}_${topic}_w2"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_progressive_pca_router_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --router_path "${routers[$slot]}" \
      --full_root "${FULL_ROOT}" \
      --output_dir "${RUN_ROOT}/${name}" \
      --topics "${topic}" \
      --window_indices 2 \
      --projection_dim 32 \
      --gqa_candidate_mode shared_mean \
      >"${RUN_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} name=${name}"
done
