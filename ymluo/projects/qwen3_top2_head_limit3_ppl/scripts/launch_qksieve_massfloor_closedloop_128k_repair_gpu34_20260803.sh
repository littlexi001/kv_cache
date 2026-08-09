#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massfloor_closedloop_128k_repair_gpu34_v1}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/home/fdong/ymluo/datasets/sklearn}"
VARIANTS="${VARIANTS:-qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor900_fulltopk_k1280,qksieve_keymse_requestlocal_valuesketch16i4_wometric_massfloor950_fulltopk_k1280}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA=1.0
export QKSIEVE_PROFILE_STAGES="${QKSIEVE_PROFILE_STAGES:-1}"
export QKSIEVE_RECORD_LAYER_CANDIDATE_STATS="${QKSIEVE_RECORD_LAYER_CANDIDATE_STATS:-1}"

mkdir -p "${OUTPUT}"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES=3,4 "${PYTHON}" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${OUTPUT}" \
  --history_tokens 131008 \
  --stream_reference_history_tokens 131008 \
  --eval_tokens 4 \
  --topic computer \
  --seed 20260833 \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --protect_recent_tokens 0 \
  --dataset_cache_dir "${DATASET_CACHE_DIR}" \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANTS}" \
  >"${OUTPUT}/run.log" 2>&1
touch "${OUTPUT}/ALL_COMPLETE"
