#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massladder_smoke_gpu1_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "${OUTPUT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${OUTPUT}" \
  --history_tokens 3968 \
  --eval_tokens 32 \
  --topic religion \
  --seed 20260861 \
  --repeat_topic_stream_if_short \
  --prefill_chunk_tokens 1024 \
  --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --max_memory_per_gpu_gib 22 \
  --variants \
    qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280,qksieve_keymse_requestlocal_valuesketch16i4_massladder90 \
  >"${OUTPUT}/run.log" 2>&1
touch "${OUTPUT}/ALL_COMPLETE"
