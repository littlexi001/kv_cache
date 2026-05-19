#!/usr/bin/env bash
set -euo pipefail

CKPT_FILE="${CKPT_FILE:-../checkpoints/inverse-kv-local-h128-l3-top1/5000.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/attention_output_probe_step5000.json}"

python3 analyze_attention_output_probe.py \
  --ckpt_file "$CKPT_FILE" \
  --output_path "$OUTPUT_PATH" \
  --seq_len "${SEQ_LEN:-128}" \
  --num_train_samples "${NUM_TRAIN_SAMPLES:-256}" \
  --num_test_samples "${NUM_TEST_SAMPLES:-128}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE:-4}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}" \
  --synthetic_seed "${SYNTHETIC_SEED:-0}" \
  --synthetic_min_token_id "${SYNTHETIC_MIN_TOKEN_ID:-1}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}" \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA:-1.0}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE:-257}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE:-128}" \
  --debug_intermediate_size "${DEBUG_INTERMEDIATE_SIZE:-256}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS:-3}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS:-4}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS:-2}" \
  --debug_head_dim "${DEBUG_HEAD_DIM:-32}" \
  --debug_max_position_embeddings "${DEBUG_MAX_POSITION_EMBEDDINGS:-256}" \
  --use_moe \
  --moe_num_unique_experts "${MOE_NUM_UNIQUE_EXPERTS:-4}" \
  --moe_num_experts_per_tok "${MOE_NUM_EXPERTS_PER_TOK:-1}" \
  --moe_intermediate_size "${MOE_INTERMEDIATE_SIZE:-128}" \
  --moe_router_input "${MOE_ROUTER_INPUT:-hidden}" \
  ${MOE_HEAD_LEVEL:+--moe_head_level} \
  --probe_epochs "${PROBE_EPOCHS:-60}" \
  --probe_lr "${PROBE_LR:-0.05}" \
  --max_probe_tokens "${MAX_PROBE_TOKENS:-32768}"
