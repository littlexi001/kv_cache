#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
INT2_ROOT="${INT2_ROOT:-${ROOT}/results/20260715_int2_scanonly_quality_32k_w0}"
RESCUE_ROOT="${RESCUE_ROOT:-${ROOT}/results/20260715_layer_rescue_early16_late2pct_32k_w0}"

mkdir -p "${INT2_ROOT}" "${RESCUE_ROOT}"
topics=(sports medicine sports medicine sports medicine)
fractions=(0.005 0.005 0.01 0.01 0.02 0.02)
gpus=(0 1 2 3 4 5)

for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  fraction="${fractions[$index]}"
  gpu="${gpus[$index]}"
  name="${topic}_w0_f${fraction}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${INT2_ROOT}/${name}" \
      --topics "${topic}" \
      --window_indices 0 \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --mass_thresholds 0.000001 \
      --budget_fractions "${fraction}" \
      --mass_estimator qabs_sampled_tail \
      --sample_fraction 0.0025 \
      --qabs_dim_count 16 \
      --candidate_fraction "${fraction}" \
      --qabs_use_cuda_kernels \
      --qabs_skip_candidate_rerank \
      --qabs_int2_onthefly \
      --prefill_chunk_tokens 2048 \
      >"${INT2_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched int2 pid=$! gpu=${gpu} topic=${topic} fraction=${fraction}"
done

rescue_topics=(sports medicine)
rescue_gpus=(6 7)
for index in "${!rescue_topics[@]}"; do
  topic="${rescue_topics[$index]}"
  gpu="${rescue_gpus[$index]}"
  name="${topic}_w0"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RESCUE_ROOT}/${name}" \
      --topics "${topic}" \
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
      --qabs_early_layer_count 16 \
      --qabs_early_budget_fraction 0.005 \
      --prefill_chunk_tokens 2048 \
      >"${RESCUE_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched rescue pid=$! gpu=${gpu} topic=${topic}"
done
