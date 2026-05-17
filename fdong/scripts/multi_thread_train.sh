#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SEQ_LEN="${SEQ_LEN:-1024}"
DATA_SHUFFLE="${DATA_SHUFFLE:-true}"  # true/false
USE_BF16="${USE_BF16:-true}"  # true/false

OPTIMIZER="${OPTIMIZER:-AdamW}"  # AdamW/sgd
LR="${LR:-1e-4}"

NUM_WORKERS="${NUM_WORKERS:-4}"
CONFIG_DIR="${CONFIG_DIR:-../Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-.}"
DATASET_TYPE="${DATASET_TYPE:-hierarchical_pattern}"  # jsonl/pruned/synthetic_indexed/hierarchical_pattern

SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-100000}"
SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-2048}"
SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-256}"
SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
SYNTHETIC_PAD_TOKEN_ID="${SYNTHETIC_PAD_TOKEN_ID:-0}"
SYNTHETIC_MIN_TOKEN_ID="${SYNTHETIC_MIN_TOKEN_ID:-1}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}"  # uniform/zipf
SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:--1}"
DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:--1}"
DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:--1}"
DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:--1}"
DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:--1}"
DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:--1}"
DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:--1}"
DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:--1}"

USE_MOE="${USE_MOE:-false}"  # true/false
MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}"
MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-1}"
MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:--1}"
MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-false}"  # true/false
MOE_COMMON_INTERMEDIATE_SIZE="${MOE_COMMON_INTERMEDIATE_SIZE:--1}"
MOE_ROUTER_INPUT="${MOE_ROUTER_INPUT:-hidden}"  # hidden/attention_output
MOE_HEAD_LEVEL="${MOE_HEAD_LEVEL:-false}"  # true/false

RUN_NAME="${RUN_NAME:-hierarchical-pattern}"

ATTENTION_STRIDE_PATTERN="${ATTENTION_STRIDE_PATTERN:-}"
RESIDUAL_SOURCE_PATTERN="${RESIDUAL_SOURCE_PATTERN:-}"


# ========== 使用 RUN_NAME 构建路径和文件名 ==========
CKPT_DIR="../checkpoints/${RUN_NAME}"

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
if [ -n "$ATTENTION_STRIDE_PATTERN" ]; then
  ARGS+=" --attention_stride_pattern=${ATTENTION_STRIDE_PATTERN}"
fi
if [ -n "$RESIDUAL_SOURCE_PATTERN" ]; then
  ARGS+=" --residual_source_pattern=${RESIDUAL_SOURCE_PATTERN}"
fi


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


mkdir -p ../logs

nohup python3 -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --master_addr=localhost \
    --master_port="${MASTER_PORT:-12345}" \
    multi_thread_train.py ${ARGS}\
    >>../logs/${RUN_NAME}.log 2>&1 &
