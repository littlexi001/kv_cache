#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/Qwen3-0.6B}"
TEXT_PATH="${TEXT_PATH:-/mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/ppl_only_20000_5000_v7}"

python "${PROJECT_DIR}/src/evaluate_qwen3_ppl_only.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --text_path "${TEXT_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --prefill_tokens "${PREFILL_TOKENS:-20000}" \
  --eval_tokens "${EVAL_TOKENS:-5000}" \
  --eval_last_tokens_only "${EVAL_LAST_TOKENS_ONLY:-false}" \
  --chunk_size "${CHUNK_SIZE:-128}" \
  --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-256}" \
  --eval_chunk_size "${EVAL_CHUNK_SIZE:-1}" \
  --max_chars "${MAX_CHARS:-8000000}" \
  --add_special_tokens "${ADD_SPECIAL_TOKENS:-false}" \
  --append_eos "${APPEND_EOS:-false}" \
  --require_total_tokens "${REQUIRE_TOTAL_TOKENS:-true}" \
  --dtype "${DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}" \
  --device_map "${DEVICE_MAP:-auto}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --compute_baseline_ppl "${COMPUTE_BASELINE_PPL:-true}" \
  --compute_tree_ppl "${COMPUTE_TREE_PPL:-true}" \
  --tree_prefill "${TREE_PREFILL:-false}" \
  --share_prefill_for_eval "${SHARE_PREFILL_FOR_EVAL:-true}" \
  --layers "${LAYERS:-all}" \
  --kv_heads "${KV_HEADS:-all}" \
  --boundary_fraction "${BOUNDARY_FRACTION:-0.005}" \
  --leaf_fraction "${LEAF_FRACTION:-0.001}" \
  --leaf_size "${LEAF_SIZE:-0}" \
  --tree_fanout "${TREE_FANOUT:-10}" \
  --tree_branch_counts "${TREE_BRANCH_COUNTS:-5,2,2}" \
  --candidate_granularity "${CANDIDATE_GRANULARITY:-layer_shared}" \
  --tree_attention_impl "${TREE_ATTENTION_IMPL:-shared_matmul}" \
  --profile_tree_stages "${PROFILE_TREE_STAGES:-true}"
