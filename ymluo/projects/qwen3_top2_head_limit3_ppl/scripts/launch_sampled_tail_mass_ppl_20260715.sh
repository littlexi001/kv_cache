#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_sampled_tail_mass_32k_w3}"

mkdir -p "${RUN_ROOT}"
topics=(sports sports sports medicine medicine medicine)
windows=(0 1 2 0 1 2)
gpus=(1 2 3 4 5 6)

for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  window="${windows[$index]}"
  gpu="${gpus[$index]}"
  name="${topic}_w${window}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    "${PYTHON_BIN}" "${ROOT}/src/run_adaptive_mass_budget_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RUN_ROOT}/${name}" \
      --topics "${topic}" \
      --window_indices "${window}" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --mass_thresholds 0.75,0.80,0.85,0.90 \
      --budget_fractions 0.0025,0.005,0.01,0.02,0.04 \
      --mass_estimator sampled_tail \
      --sample_fraction 0.0025 \
      --prefill_chunk_tokens 2048 \
      >"${RUN_ROOT}/${name}.log" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} topic=${topic} window=${window}"
  sleep 1
done
