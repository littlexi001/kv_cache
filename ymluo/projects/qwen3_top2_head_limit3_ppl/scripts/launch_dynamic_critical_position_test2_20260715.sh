#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
ROUTER="${ROUTER:-${ROOT}/results/20260715_critical_position_router_train01_test2/critical_position_router.pkl}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260715_dynamic_critical_position_test2}"

mkdir -p "${RUN_ROOT}"

topics=(sports medicine)
gpus=(6 7)

for index in "${!topics[@]}"; do
  topic="${topics[$index]}"
  gpu="${gpus[$index]}"
  output_dir="${RUN_ROOT}/${topic}_w2"
  log_path="${RUN_ROOT}/${topic}_w2.log"

  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${ROOT}/src" \
    "${PYTHON_BIN}" "${ROOT}/src/run_dynamic_critical_position_ppl_20260715.py" \
      --model_name_or_path "${MODEL}" \
      --router_path "${ROUTER}" \
      --output_dir "${output_dir}" \
      --topics "${topic}" \
      --window_indices 2 \
      --history_tokens 32000 \
      --query_tokens 256 \
      --eval_tokens 256 \
      --window_stride_tokens 32512 \
      --prefill_chunk_tokens 2048 \
      >"${log_path}" 2>&1 </dev/null &
  echo "launched pid=$! gpu=${gpu} topic=${topic} log=${log_path}"
  sleep 1
done
