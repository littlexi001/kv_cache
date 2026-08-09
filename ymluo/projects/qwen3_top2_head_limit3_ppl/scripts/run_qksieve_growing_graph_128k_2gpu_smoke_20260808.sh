#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
OUTPUT="${OUTPUT:-${ROOT}/results/model_growing_graph_qk_128k_2gpu_smoke_20260808}"
GPU_PAIR="${GPU_PAIR:-0,1}"
VARIANT="${VARIANT:-qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280}"
HISTORY="${HISTORY:-131072}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

export QKSIEVE_QK_FACTOR_SOLVER=legacy
export QKSIEVE_QMSE_RATE_ALLOCATOR=torch
export QKSIEVE_PRELOAD_EXTENSIONS=1
export QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
export QKSIEVE_BUILD_RESIDENT_VALUE_SKETCH=1
export QKSIEVE_RESIDENT_VALUE_WORKERS=12
export QKSIEVE_PARALLEL_QK_WORKERS=36
export QKSIEVE_PARALLEL_VALUE_WORKERS=0
export QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
export QKSIEVE_BATCH_QMSE_ALLOCATION="${QKSIEVE_BATCH_QMSE_ALLOCATION:-1}"
export QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
export QKSIEVE_TILED_VALUE_ATTENTION=0
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0

export QKSIEVE_MODEL_GROWING_CUDAGRAPH_PROBE="${ENABLE_GROWING_GRAPH:-1}"
export QKSIEVE_MODEL_GROWING_CUDAGRAPH_METHOD=direct_countcap
export QKSIEVE_MODEL_GROWING_GRAPH_CORRECTNESS_STEPS="${CORRECTNESS_STEPS:-2}"
export QKSIEVE_MODEL_GROWING_GRAPH_WARMUP_STEPS="${WARMUP_STEPS:-2}"
export QKSIEVE_MODEL_GROWING_GRAPH_TIMING_STEPS="${TIMING_STEPS:-5}"
export QKSIEVE_GRAPH_SUFFIX_CAPACITY="${SUFFIX_CAPACITY:-32}"

mkdir -p "${OUTPUT}"
EXTRA_ARGS=()
if [[ "${FULL_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--full_only)
fi
if [[ "${USE_TEXT_FILE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--text_file "${TEXT_FILE_PATH:-${ROOT}/src/run_head_top2_targeted_ppl_20260714.py}")
fi
exec env CUDA_VISIBLE_DEVICES="${GPU_PAIR}" "${PYTHON}" -u \
  "${ROOT}/src/run_qksieve_coldskip_longcontext_quality_20260730.py" \
  --model_name_or_path "${MODEL}" \
  --template "${ROOT}/nonexistent_requestlocal_template.pt" \
  --output_dir "${OUTPUT}" \
  --history_tokens "${HISTORY}" \
  --stream_reference_history_tokens "${HISTORY}" \
  --eval_tokens "${EVAL_TOKENS:-2}" \
  --topic "${TOPIC:-mixed_b}" \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --protect_recent_tokens 0 \
  --dataset_cache_dir "${ROOT}/datasets" \
  --seed 20260808 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANT}" \
  "${EXTRA_ARGS[@]}"
