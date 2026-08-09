#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON_BIN="${PYTHON_BIN:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/results/20260725_direct_countcap_denseprompt_ppl_32k_4gpu}"
RUNNER="${PROJECT_DIR}/src/run_direct_countcap_denseprompt_ppl_20260725.py"

mkdir -p "${RUN_ROOT}/logs"

launch() {
  local gpu="$1"
  local topic="$2"
  local methods="$3"
  local name="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RUNNER}" \
    --model_name_or_path "${MODEL_PATH}" \
    --output_dir "${RUN_ROOT}/${name}" \
    --topics "${topic}" \
    --window_indices 0,1,2 \
    --methods "${methods}" \
    --history_tokens 32000 \
    --eval_tokens 256 \
    --window_stride_tokens 32512 \
    --direct_fraction 0.06 \
    --direct_min_tokens 256 \
    --direct_max_tokens 1280 \
    --projection_dim 48 \
    --sample_count 256 \
    --prefill_chunk_tokens 2048 \
    --dataset_cache_dir "${DATASET_CACHE_DIR}" \
    >"${RUN_ROOT}/logs/${name}.log" 2>&1 &
  echo "$!" >"${RUN_ROOT}/logs/${name}.pid"
  echo "[launch] gpu=${gpu} pid=$! name=${name}"
}

launch 0 sports direct_countcap sports_direct
launch 1 medicine direct_countcap medicine_direct
launch 2 sports full_attention,exact_top2 sports_reference
launch 3 medicine full_attention,exact_top2 medicine_reference

wait
touch "${RUN_ROOT}/ALL_COMPLETE"
echo "[complete] ${RUN_ROOT}"
