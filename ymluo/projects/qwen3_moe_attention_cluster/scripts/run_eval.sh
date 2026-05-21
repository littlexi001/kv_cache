#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/eval_attention_cluster.py" \
  --config_dir "${CONFIG_DIR:-${REPO_ROOT}/fdong/Qwen3-0.6B}" \
  --ckpt_file "${CKPT_FILE:-${PROJECT_DIR}/outputs/train/moe-attention-cluster/checkpoints/10000.pth}" \
  --output_path "${OUTPUT_PATH:-}" \
  --seed "${SEED:-1234}" \
  --device "${DEVICE:-cuda:0}" \
  --use_bf16 "${USE_BF16:-false}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --seq_len "${SEQ_LEN:-128}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --eval_batches "${EVAL_BATCHES:-32}" \
  --synthetic_num_samples "${SYNTHETIC_NUM_SAMPLES:-200000}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE:-4}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}" \
  --synthetic_seed "${SYNTHETIC_SEED:-0}" \
  --synthetic_pad_token_id "${SYNTHETIC_PAD_TOKEN_ID:-0}" \
  --synthetic_min_token_id "${SYNTHETIC_MIN_TOKEN_ID:-1}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}" \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA:-1.0}" \
  --synthetic_zipf_shuffle_ranks "${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE:-257}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE:-128}" \
  --debug_intermediate_size "${DEBUG_INTERMEDIATE_SIZE:-256}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS:-3}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS:-4}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS:-2}" \
  --debug_head_dim "${DEBUG_HEAD_DIM:-32}" \
  --debug_max_position_embeddings "${DEBUG_MAX_POSITION_EMBEDDINGS:-256}" \
  --attention_stride_pattern "${ATTENTION_STRIDE_PATTERN:-}" \
  --residual_source_pattern "${RESIDUAL_SOURCE_PATTERN:-}" \
  --use_moe "${USE_MOE:-true}" \
  --moe_num_unique_experts "${MOE_NUM_UNIQUE_EXPERTS:-4}" \
  --moe_num_experts_per_tok "${MOE_NUM_EXPERTS_PER_TOK:-1}" \
  --moe_intermediate_size "${MOE_INTERMEDIATE_SIZE:-128}" \
  --moe_use_common_expert "${MOE_USE_COMMON_EXPERT:-false}" \
  --moe_common_intermediate_size "${MOE_COMMON_INTERMEDIATE_SIZE:--1}" \
  --moe_router_bias "${MOE_ROUTER_BIAS:-false}" \
  --moe_normalize_topk_prob "${MOE_NORMALIZE_TOPK_PROB:-true}" \
  --moe_router_input "${MOE_ROUTER_INPUT:-attention_output}" \
  --moe_head_level "${MOE_HEAD_LEVEL:-false}" \
  "$@"
