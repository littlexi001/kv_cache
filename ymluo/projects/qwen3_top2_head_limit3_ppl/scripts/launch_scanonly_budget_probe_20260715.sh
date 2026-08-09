#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_scanonly_budget_probe_32k_w0}"

mkdir -p "${RUN_ROOT}"
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
      --output_dir "${RUN_ROOT}/${name}" \
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
      --prefill_chunk_tokens 2048 \
      >"${RUN_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} topic=${topic} fraction=${fraction}"
done
