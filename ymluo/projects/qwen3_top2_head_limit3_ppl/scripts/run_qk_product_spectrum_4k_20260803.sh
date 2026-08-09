#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260803_qk_product_spectrum_llama31_8b_4k}"
GPU_ID="${GPU_ID:-0}"
TRACE="${RUN_ROOT}/traces/llama31_8b_sports4k.pt"
LAYERS="${LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RUN_ROOT}/traces" "${RUN_ROOT}/analysis" "${RUN_ROOT}/logs"
cd "${ROOT}"

if [[ ! -s "${TRACE}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "${MODEL}" \
    --output_path "${TRACE}" \
    --topic sports \
    --history_tokens 4096 \
    --steps 8 \
    --layers "${LAYERS}" \
    --prefill_chunk_tokens 1024 \
    --omit_values \
    --seed 20260803 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"${RUN_ROOT}/logs/capture.log" 2>&1
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -u \
  src/analyze_qk_product_spectrum_20260803.py \
  --trace_path "${TRACE}" \
  --output_dir "${RUN_ROOT}/analysis" \
  --label llama31_8b_sports4k \
  --sample_stride 32 \
  --calibration_steps 8 \
  --query_shrinkages 0,0.75 \
  --device cuda \
  >"${RUN_ROOT}/logs/analysis.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
