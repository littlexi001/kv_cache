#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/attention_pruning_cos_ppl}"

python "${PROJECT_DIR}/src/analyze_qwen3_attention_pruning_cos_ppl.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-5000}" \
  --eval_tokens "${EVAL_TOKENS:-5000}" \
  --eval_last_tokens_only "${EVAL_LAST_TOKENS_ONLY:-false}" \
  --chunk_size "${CHUNK_SIZE:-256}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_total_tokens "${REQUIRE_TOTAL_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --layers "${LAYERS:-all}" \
  --heads "${HEADS:-all}" \
  --ratios "${RATIOS:-0.001,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.15,0.20}" \
  --save_cos_per_token "${SAVE_COS_PER_TOKEN:-true}" \
  --cos_csv_flush_rows "${COS_CSV_FLUSH_ROWS:-200000}" \
  --compute_cos "${COMPUTE_COS:-true}" \
  --compute_ppl "${COMPUTE_PPL:-true}" \
  --make_plots "${MAKE_PLOTS:-true}" \
  --plot_dpi "${PLOT_DPI:-180}"
