#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SEQ_LEN="${SEQ_LEN:-64}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"  # true/false
USE_BF16="${USE_BF16:-false}"  # true/false

OPTIMIZER="${OPTIMIZER:-AdamW}"  # AdamW/sgd
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"

NUM_WORKERS="${NUM_WORKERS:-0}"
CONFIG_DIR="${CONFIG_DIR:-../Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-.}"
DATASET_TYPE="${DATASET_TYPE:-hierarchical_pattern}"  # jsonl/pruned/synthetic_indexed/hierarchical_pattern

SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-128}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-128}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-32}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_PAD_TOKEN_ID="${SYNTHETIC_PAD_TOKEN_ID:-0}"
SYNTHETIC_MIN_TOKEN_ID="${SYNTHETIC_MIN_TOKEN_ID:-1}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}"  # uniform/zipf
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-256}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-64}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-128}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-2}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-16}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

USE_MOE="${USE_MOE:-true}"  # true/false
MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}"
MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-2}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-64}"
MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-true}"  # true/false
MOE_COMMON_INTERMEDIATE_SIZE="${MOE_COMMON_INTERMEDIATE_SIZE:-64}"
MOE_ROUTER_INPUT="${MOE_ROUTER_INPUT:-hidden}"  # hidden/attention_output
MOE_HEAD_LEVEL="${MOE_HEAD_LEVEL:-false}"  # true/false
MOE_LOAD_BALANCE_LOSS_WEIGHT="${MOE_LOAD_BALANCE_LOSS_WEIGHT:-0.0}"

ATTENTION_STRIDE_PATTERN="${ATTENTION_STRIDE_PATTERN-1,4}"
RESIDUAL_SOURCE_PATTERN="${RESIDUAL_SOURCE_PATTERN--1,-1}"


# ========== 动态构建 CKPT_DIR ==========
CKPT_DIR="${CKPT_DIR:-../checkpoints/single-thread-debug}"

# ========== 构建 Python 命令 ==========
ARGS=""
ARGS+=" --local_batch_size $LOCAL_BATCH_SIZE"
ARGS+=" --global_batch_size $GLOBAL_BATCH_SIZE"
ARGS+=" --save_interval $SAVE_INTERVAL"
ARGS+=" --seq_len $SEQ_LEN"
ARGS+=" --num_workers $NUM_WORKERS"
ARGS+=" --config_dir $CONFIG_DIR"
ARGS+=" --data_dir $DATA_DIR"
ARGS+=" --dataset_type $DATASET_TYPE"
ARGS+=" --ckpt_dir $CKPT_DIR"  # ← 关键：传入构建好的路径
ARGS+=" --synthetic_num_samples $SYNTHETIC_NUM_SAMPLES"
ARGS+=" --synthetic_block_size $SYNTHETIC_BLOCK_SIZE"
ARGS+=" --synthetic_num_hierarchy_layers $SYNTHETIC_NUM_HIERARCHY_LAYERS"
ARGS+=" --synthetic_content_token_count $SYNTHETIC_CONTENT_TOKEN_COUNT"
ARGS+=" --synthetic_num_units_per_layer $SYNTHETIC_NUM_UNITS_PER_LAYER"
ARGS+=" --synthetic_seed $SYNTHETIC_SEED"
ARGS+=" --synthetic_pad_token_id $SYNTHETIC_PAD_TOKEN_ID"
ARGS+=" --synthetic_min_token_id $SYNTHETIC_MIN_TOKEN_ID"
ARGS+=" --synthetic_sampling_distribution $SYNTHETIC_SAMPLING_DISTRIBUTION"
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
if [ "$USE_MOE" = "true" ]; then
  ARGS+=" --use_moe"
fi
ARGS+=" --moe_num_unique_experts $MOE_NUM_UNIQUE_EXPERTS"
ARGS+=" --moe_num_experts_per_tok $MOE_NUM_EXPERTS_PER_TOK"
ARGS+=" --moe_intermediate_size $MOE_INTERMEDIATE_SIZE"
if [ "$MOE_USE_COMMON_EXPERT" = "true" ]; then
  ARGS+=" --moe_use_common_expert"
fi
ARGS+=" --moe_common_intermediate_size $MOE_COMMON_INTERMEDIATE_SIZE"
ARGS+=" --moe_router_input $MOE_ROUTER_INPUT"
if [ "$MOE_HEAD_LEVEL" = "true" ]; then
  ARGS+=" --moe_head_level"
fi
ARGS+=" --moe_load_balance_loss_weight $MOE_LOAD_BALANCE_LOSS_WEIGHT"
ARGS+=" --attention_stride_pattern=${ATTENTION_STRIDE_PATTERN}"
ARGS+=" --residual_source_pattern=${RESIDUAL_SOURCE_PATTERN}"


# 处理布尔参数
if [ "$DATA_SHUFFLE" = "true" ]; then
  ARGS+=" --data_shuffle"
else
  ARGS+=" --no_data_shuffle"
fi

if [ "$USE_BF16" = "true" ]; then
  ARGS+=" --use_bf16"
else
  ARGS+=" --no_use_bf16"
fi

ARGS+=" --optimizer $OPTIMIZER"
ARGS+=" --lr $LR"
ARGS+=" --warmup_steps $WARMUP_STEPS"
ARGS+=" --total_training_steps $TOTAL_TRAINING_STEPS"

python3 single_thread_debug.py ${ARGS}
