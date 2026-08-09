#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260804_qksieve_v37_direct_decode_length_4gpu_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
unset QKSIEVE_PROFILE_STAGES || true
unset QKSIEVE_EXACT_SELECTION_DIAGNOSTICS || true

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/n65536"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${PYTHON}" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${RUN_ROOT}/n65536" \
  --history_tokens 65536 \
  --stream_reference_history_tokens 65536 \
  --eval_tokens "${EVAL_TOKENS:-32}" \
  --topic mixed_b \
  --seed 20261864 \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --protect_recent_tokens 0 \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants qksieve_qmse_oas_requestlocal_valuesketch16_k1280

touch "${RUN_ROOT}/n65536/ALL_COMPLETE"
