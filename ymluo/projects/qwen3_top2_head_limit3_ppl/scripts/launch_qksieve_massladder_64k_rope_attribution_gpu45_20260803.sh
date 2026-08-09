#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
OUTPUT="${OUTPUT:-${ROOT}/results/20260803_qksieve_massladder_64k_rope_attribution_gpu45_v1}"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
export QKSIEVE_MASS_LADDER_FLOOR_K=1280
export QKSIEVE_MASS_LADDER_GROWTH=1.5
export QKSIEVE_MASS_LADDER_MAX_FRACTION=0.25

mkdir -p "${OUTPUT}"
cd "${ROOT}"

VARIANTS="exact_qk_oracle_k1280,exact_qk_oracle_k2560,exact_qk_oracle_k5120,exact_qk_oracle_k10240"
VARIANTS+=",qksieve_keymse_requestlocal_sampled_k1280_c32,qksieve_keymse_requestlocal_sampled_k2560_c32,qksieve_keymse_requestlocal_sampled_k5120_c32,qksieve_keymse_requestlocal_sampled_k10240_c32"
VARIANTS+=",qksieve_keymse_requestlocal_valuesketch16i4_sampled_k1280,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k2560,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k5120,qksieve_keymse_requestlocal_valuesketch16i4_sampled_k10240"
VARIANTS+=",qksieve_keymse_requestlocal_valuesketch16i4_massladder90,qksieve_keymse_requestlocal_valuesketch16i4_massladder95"

CUDA_VISIBLE_DEVICES=4,5 "${PYTHON}" -u \
  src/run_qksieve_coldskip_longcontext_quality_20260730.py \
  --model_name_or_path "${MODEL}" \
  --template "${TEMPLATE}" \
  --output_dir "${OUTPUT}" \
  --history_tokens 64000 \
  --eval_tokens 2 \
  --synthetic_rope_seed 20260881 \
  --prefill_chunk_tokens 1024 \
  --dataset_cache_dir /home/fdong/ymluo/datasets/sklearn \
  --dtype float16 \
  --device cuda \
  --device_map balanced \
  --max_memory_per_gpu_gib 22 \
  --variants "${VARIANTS}" \
  >"${OUTPUT}/run.log" 2>&1

touch "${OUTPUT}/ALL_COMPLETE"
