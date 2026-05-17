#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CONFIG_DIR="${CONFIG_DIR:-../Qwen3-0.6B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-../checkpoints}"
RUN_SPECS="${RUN_SPECS:-uniform_baseline:inverse-kv-local-h128-l3-top1:uniform,zipf_baseline:inverse-kv-zipf-baseline:zipf}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/dominant_directions_baselines_step${CHECKPOINT_STEP}.json}"

SEQ_LEN="${SEQ_LEN:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"

SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_MIN_TOKEN_ID="${SYNTHETIC_MIN_TOKEN_ID:-1}"
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.1}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-257}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-128}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-256}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-3}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-32}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

TOP_TOKENS="${TOP_TOKENS:-8}"

ARGS=""
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --checkpoint_root $CHECKPOINT_ROOT"
ARGS+=" --run_specs $RUN_SPECS"
ARGS+=" --checkpoint_step $CHECKPOINT_STEP"
ARGS+=" --output_path $OUTPUT_PATH"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --num_samples $NUM_SAMPLES"
ARGS+=" --batch_size $BATCH_SIZE"
ARGS+=" --synthetic_block_size $SYNTHETIC_BLOCK_SIZE"
ARGS+=" --synthetic_num_hierarchy_layers $SYNTHETIC_NUM_HIERARCHY_LAYERS"
ARGS+=" --synthetic_content_token_count $SYNTHETIC_CONTENT_TOKEN_COUNT"
ARGS+=" --synthetic_num_units_per_layer $SYNTHETIC_NUM_UNITS_PER_LAYER"
ARGS+=" --synthetic_seed $SYNTHETIC_SEED"
ARGS+=" --synthetic_min_token_id $SYNTHETIC_MIN_TOKEN_ID"
ARGS+=" --synthetic_zipf_alpha $SYNTHETIC_ZIPF_ALPHA"
if [ "$SYNTHETIC_ZIPF_SHUFFLE_RANKS" = "true" ]; then
  ARGS+=" --synthetic_zipf_shuffle_ranks"
else
  ARGS+=" --synthetic_no_zipf_shuffle_ranks"
fi
ARGS+=" --debug_vocab_size $DEBUG_VOCAB_SIZE"
ARGS+=" --debug_hidden_size $DEBUG_HIDDEN_SIZE"
ARGS+=" --debug_intermediate_size $DEBUG_INTERMEDIATE_SIZE"
ARGS+=" --debug_num_hidden_layers $DEBUG_NUM_HIDDEN_LAYERS"
ARGS+=" --debug_num_attention_heads $DEBUG_NUM_ATTENTION_HEADS"
ARGS+=" --debug_num_key_value_heads $DEBUG_NUM_KEY_VALUE_HEADS"
ARGS+=" --debug_head_dim $DEBUG_HEAD_DIM"
ARGS+=" --debug_max_position_embeddings $DEBUG_MAX_POSITION_EMBEDDINGS"
ARGS+=" --top_tokens $TOP_TOKENS"

python3 analyze_dominant_directions.py ${ARGS}
