#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/eval_topk_negative_update.py" \
  --config_dir "${CONFIG_DIR:-${REPO_ROOT}/fdong/Qwen3-0.6B}" \
  --ckpt_file "${CKPT_FILE:-${PROJECT_DIR}/outputs/train/topk-negative-structured/checkpoints/10000.pth}" \
  --output_path "${OUTPUT_PATH:-}" \
  --seed "${SEED:-1234}" \
  --device "${DEVICE:-cuda:0}" \
  --use_bf16 "${USE_BF16:-false}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --seq_len "${SEQ_LEN:-256}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}" \
  --eval_batches "${EVAL_BATCHES:-32}" \
  --synthetic_data_mode "${SYNTHETIC_DATA_MODE:-structured_language}" \
  --synthetic_num_samples "${SYNTHETIC_NUM_SAMPLES:-200000}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE:-4}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}" \
  --synthetic_seed "${SYNTHETIC_SEED:-0}" \
  --synthetic_pad_token_id "${SYNTHETIC_PAD_TOKEN_ID:-0}" \
  --synthetic_min_token_id "${SYNTHETIC_MIN_TOKEN_ID:-1}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}" \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA:-1.1}" \
  --synthetic_zipf_shuffle_ranks "${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}" \
  --structured_topic_count "${STRUCTURED_TOPIC_COUNT:-8}" \
  --structured_entities_per_topic "${STRUCTURED_ENTITIES_PER_TOPIC:-8}" \
  --structured_shared_entity_count "${STRUCTURED_SHARED_ENTITY_COUNT:-16}" \
  --structured_verb_count "${STRUCTURED_VERB_COUNT:-12}" \
  --structured_function_token_count "${STRUCTURED_FUNCTION_TOKEN_COUNT:-12}" \
  --structured_noise_token_count "${STRUCTURED_NOISE_TOKEN_COUNT:-32}" \
  --structured_topic_zipf_alpha "${STRUCTURED_TOPIC_ZIPF_ALPHA:-1.1}" \
  --structured_noise_rate "${STRUCTURED_NOISE_RATE:-0.25}" \
  --structured_ambiguity_rate "${STRUCTURED_AMBIGUITY_RATE:-0.35}" \
  --structured_copy_rate "${STRUCTURED_COPY_RATE:-0.25}" \
  --structured_bridge_rate "${STRUCTURED_BRIDGE_RATE:-0.25}" \
  --structured_min_span_units "${STRUCTURED_MIN_SPAN_UNITS:-2}" \
  --structured_max_span_units "${STRUCTURED_MAX_SPAN_UNITS:-8}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE:-512}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE:-128}" \
  --debug_intermediate_size "${DEBUG_INTERMEDIATE_SIZE:-256}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS:-3}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS:-4}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS:-2}" \
  --debug_head_dim "${DEBUG_HEAD_DIM:-32}" \
  --debug_max_position_embeddings "${DEBUG_MAX_POSITION_EMBEDDINGS:-512}" \
  --attention_stride_pattern "${ATTENTION_STRIDE_PATTERN:-}" \
  --residual_source_pattern "${RESIDUAL_SOURCE_PATTERN:-}" \
  --use_moe "${USE_MOE:-true}" \
  --moe_num_unique_experts "${MOE_NUM_UNIQUE_EXPERTS:-16}" \
  --moe_num_experts_per_tok "${MOE_NUM_EXPERTS_PER_TOK:-4}" \
  --moe_intermediate_size "${MOE_INTERMEDIATE_SIZE:-128}" \
  --moe_use_common_expert "${MOE_USE_COMMON_EXPERT:-false}" \
  --moe_common_intermediate_size "${MOE_COMMON_INTERMEDIATE_SIZE:--1}" \
  --moe_router_bias "${MOE_ROUTER_BIAS:-false}" \
  --moe_normalize_topk_prob "${MOE_NORMALIZE_TOPK_PROB:-true}" \
  --moe_router_input "${MOE_ROUTER_INPUT:-attention_output}" \
  --moe_head_level "${MOE_HEAD_LEVEL:-false}" \
  --use_pre_router "${USE_PRE_ROUTER:-true}" \
  --pre_router_input "${PRE_ROUTER_INPUT:-q}" \
  --pre_router_controls_attention "${PRE_ROUTER_CONTROLS_ATTENTION:-false}" \
  --moe_expert_input_attention_topk "${MOE_EXPERT_INPUT_ATTENTION_TOPK:-0}" \
  --negative_update_secondaries "${NEGATIVE_UPDATE_SECONDARIES:-true}" \
  --negative_update_scale "${NEGATIVE_UPDATE_SCALE:-1.0}" \
  --negative_update_primary_slots "${NEGATIVE_UPDATE_PRIMARY_SLOTS:-1}" \
  "$@"
