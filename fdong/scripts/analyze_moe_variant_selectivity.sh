#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONFIG_DIR="${CONFIG_DIR:-../Qwen3-0.6B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-../checkpoints}"
RUNS="${RUNS:-inverse-kv-local-h128-l3-top1,inverse-kv-attn-output-router,inverse-kv-head-moe-hidden-router,inverse-kv-attn-output-head-moe}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/moe_variant_selectivity_step${CHECKPOINT_STEP}.json}"

SEQ_LEN="${SEQ_LEN:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"

SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_MIN_TOKEN_ID="${SYNTHETIC_MIN_TOKEN_ID:-1}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}"  # uniform/zipf
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-257}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-128}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-256}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-3}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-32}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

MODES="${MODES:-full,same_slot_occurrence,same_slot,same_higher,random_same_size}"  # full/same_slot_occurrence/same_slot/same_higher/same_slot_or_higher/random_same_size
RANDOM_SEED="${RANDOM_SEED:-123}"

python3 analyze_moe_variant_selectivity.py \
  --config_dir "$CONFIG_DIR" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --runs "$RUNS" \
  --checkpoint_step "$CHECKPOINT_STEP" \
  --output_path "$OUTPUT_PATH" \
  --seq_len "$SEQ_LEN" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --synthetic_block_size "$SYNTHETIC_BLOCK_SIZE" \
  --synthetic_num_hierarchy_layers "$SYNTHETIC_NUM_HIERARCHY_LAYERS" \
  --synthetic_content_token_count "$SYNTHETIC_CONTENT_TOKEN_COUNT" \
  --synthetic_num_units_per_layer "$SYNTHETIC_NUM_UNITS_PER_LAYER" \
  --synthetic_seed "$SYNTHETIC_SEED" \
  --synthetic_min_token_id "$SYNTHETIC_MIN_TOKEN_ID" \
  --synthetic_sampling_distribution "$SYNTHETIC_SAMPLING_DISTRIBUTION" \
  --synthetic_zipf_alpha "$SYNTHETIC_ZIPF_ALPHA" \
  --debug_vocab_size "$DEBUG_VOCAB_SIZE" \
  --debug_hidden_size "$DEBUG_HIDDEN_SIZE" \
  --debug_intermediate_size "$DEBUG_INTERMEDIATE_SIZE" \
  --debug_num_hidden_layers "$DEBUG_NUM_HIDDEN_LAYERS" \
  --debug_num_attention_heads "$DEBUG_NUM_ATTENTION_HEADS" \
  --debug_num_key_value_heads "$DEBUG_NUM_KEY_VALUE_HEADS" \
  --debug_head_dim "$DEBUG_HEAD_DIM" \
  --debug_max_position_embeddings "$DEBUG_MAX_POSITION_EMBEDDINGS" \
  --modes "$MODES" \
  --random_seed "$RANDOM_SEED" \
  $( [ "$SYNTHETIC_ZIPF_SHUFFLE_RANKS" = "true" ] && echo "--synthetic_zipf_shuffle_ranks" || echo "--synthetic_no_zipf_shuffle_ranks" )
