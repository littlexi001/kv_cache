#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/kvcache_svd_profile}"

python "${PROJECT_DIR}/src/profile_qwen3_kvcache_svd.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_lengths "${CACHE_LENGTHS:-1k,10k,100k,1M}" \
  --chunk_size "${CHUNK_SIZE:-512}" \
  --max_chars "${MAX_CHARS:-0}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_max_length "${REQUIRE_MAX_LENGTH:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-auto}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --cache_kinds "${CACHE_KINDS:-key,value}" \
  --percentiles "${PERCENTILES:-1,5,25,50,75,95,99}" \
  --svd_device "${SVD_DEVICE:-auto}" \
  --svd_dtype "${SVD_DTYPE:-float32}" \
  --max_svd_rank "${MAX_SVD_RANK:-128}" \
  --svd_full_matrices "${SVD_FULL_MATRICES:-false}" \
  --save_svd_tensors "${SAVE_SVD_TENSORS:-false}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_dpi "${PLOT_DPI:-160}" \
  --sample_seed "${SAMPLE_SEED:-1234}"
