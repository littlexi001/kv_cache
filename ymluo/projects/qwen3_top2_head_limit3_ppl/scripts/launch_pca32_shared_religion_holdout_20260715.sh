#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
FULL_ROOT="${ROOT}/results/20260715_full_reference_religion_holdout"
RUN_ROOT="${ROOT}/results/20260715_pca32_shared_mean_religion_holdout"
GPU="${GPU:-7}"
mkdir -p "${FULL_ROOT}" "${RUN_ROOT}"

(
  env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_critical_position_budget_probe_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${FULL_ROOT}/religion_w0" \
      --topics religion \
      --window_indices 0 \
      --only_full

  env \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RUN_ROOT}/religion_w0_f0.02" \
      --topics religion \
      --window_indices 0 \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --mass_thresholds 0.000001 \
      --budget_fractions 0.02 \
      --mass_estimator qabs_sampled_tail \
      --sample_fraction 0.0025 \
      --qabs_dim_count 16 \
      --candidate_fraction 0.02 \
      --qabs_use_cuda_kernels \
      --qabs_skip_candidate_rerank \
      --qabs_score_mode pca_int8 \
      --qabs_projection_dim 32 \
      --qabs_gqa_candidate_mode shared_mean \
      --prefill_chunk_tokens 2048
) >"${RUN_ROOT}/religion_w0_queue.log" 2>&1 </dev/null &
echo "launched queue pid=$! gpu=${GPU}"
