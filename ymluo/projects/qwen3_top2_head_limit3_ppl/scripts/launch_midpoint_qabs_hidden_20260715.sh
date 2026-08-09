#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_midpoint_qabs05_hidden_32k_w012}"

mkdir -p "${RUN_ROOT}"
topics=(sports sports sports medicine medicine medicine)
windows=(0 1 2 0 1 2)
gpus=(0 1 2 3 4 5)
pids=()

for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  window="${windows[$index]}"
  gpu="${gpus[$index]}"
  name="${topic}_w${window}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    "${PYTHON_BIN}" "${ROOT}/src/collect_critical_position_hidden_states_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --output_dir "${RUN_ROOT}/${name}" \
      --topics "${topic}" \
      --window_indices "${window}" \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --top_fraction 0.005 \
      --attention_mode qabs_scan \
      --hidden_layers 8,16,24,32 \
      --qabs_dim_count 16 \
      --qabs_use_cuda_kernels \
      --prefill_chunk_tokens 2048 \
      >"${RUN_ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
echo "all midpoint hidden jobs completed"
