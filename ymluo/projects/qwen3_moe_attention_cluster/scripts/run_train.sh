#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

print_config() {
  cat <<EOF
===== qwen3_moe_attention_cluster run_train.sh config =====
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
script=${BASH_SOURCE[0]}
project_dir=${PROJECT_DIR}
repo_root=${REPO_ROOT}
CONFIG_DIR=${CONFIG_DIR:-${REPO_ROOT}/fdong/Qwen3-0.6B}
OUT_DIR=${OUT_DIR:-${PROJECT_DIR}/outputs/train}
RUN_NAME=${RUN_NAME:-moe-attention-cluster}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
TOTAL_STEPS=${TOTAL_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
LR=${LR:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-100}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
EVAL_BATCHES=${EVAL_BATCHES:-8}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SEED=${SEED:-1234}
DEVICE=${DEVICE:-cuda:0}
USE_BF16=${USE_BF16:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
SEQ_LEN=${SEQ_LEN:-128}
SYNTHETIC_NUM_SAMPLES=${SYNTHETIC_NUM_SAMPLES:-200000}
SYNTHETIC_BLOCK_SIZE=${SYNTHETIC_BLOCK_SIZE:-4}
SYNTHETIC_NUM_HIERARCHY_LAYERS=${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}
SYNTHETIC_CONTENT_TOKEN_COUNT=${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}
SYNTHETIC_NUM_UNITS_PER_LAYER=${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}
SYNTHETIC_SEED=${SYNTHETIC_SEED:-0}
SYNTHETIC_SAMPLING_DISTRIBUTION=${SYNTHETIC_SAMPLING_DISTRIBUTION:-uniform}
DEBUG_VOCAB_SIZE=${DEBUG_VOCAB_SIZE:-257}
DEBUG_HIDDEN_SIZE=${DEBUG_HIDDEN_SIZE:-128}
DEBUG_INTERMEDIATE_SIZE=${DEBUG_INTERMEDIATE_SIZE:-256}
DEBUG_NUM_HIDDEN_LAYERS=${DEBUG_NUM_HIDDEN_LAYERS:-3}
DEBUG_NUM_ATTENTION_HEADS=${DEBUG_NUM_ATTENTION_HEADS:-4}
DEBUG_NUM_KEY_VALUE_HEADS=${DEBUG_NUM_KEY_VALUE_HEADS:-2}
DEBUG_HEAD_DIM=${DEBUG_HEAD_DIM:-32}
USE_MOE=${USE_MOE:-true}
MOE_NUM_UNIQUE_EXPERTS=${MOE_NUM_UNIQUE_EXPERTS:-4}
MOE_NUM_EXPERTS_PER_TOK=${MOE_NUM_EXPERTS_PER_TOK:-1}
MOE_INTERMEDIATE_SIZE=${MOE_INTERMEDIATE_SIZE:-128}
MOE_USE_COMMON_EXPERT=${MOE_USE_COMMON_EXPERT:-false}
MOE_ROUTER_INPUT=${MOE_ROUTER_INPUT:-attention_output}
MOE_HEAD_LEVEL=${MOE_HEAD_LEVEL:-false}
GATE_INHIBITION_WEIGHT=${GATE_INHIBITION_WEIGHT:-0.0}
ATTENTION_CLUSTER_WEIGHT=${ATTENTION_CLUSTER_WEIGHT:-0.05}
ATTENTION_CLUSTER_TEMPERATURE=${ATTENTION_CLUSTER_TEMPERATURE:-1.0}
ATTENTION_CLUSTER_TOPK=${ATTENTION_CLUSTER_TOPK:-4}
ATTENTION_CLUSTER_INCLUDE_SELF=${ATTENTION_CLUSTER_INCLUDE_SELF:-false}
ATTENTION_CLUSTER_DETACH_ATTENTION=${ATTENTION_CLUSTER_DETACH_ATTENTION:-true}
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=${ATTENTION_CLUSTER_NEGATIVE_WEIGHT:-0.01}
ATTENTION_CLUSTER_NEGATIVE_FEATURE_LAYER=${ATTENTION_CLUSTER_NEGATIVE_FEATURE_LAYER:-1}
ATTENTION_CLUSTER_NEGATIVE_HISTORY_ONLY=${ATTENTION_CLUSTER_NEGATIVE_HISTORY_ONLY:-false}
MOE_LOAD_BALANCE_LOSS_WEIGHT=${MOE_LOAD_BALANCE_LOSS_WEIGHT:-0.0}
EXPERT_REPULSION_WEIGHT=${EXPERT_REPULSION_WEIGHT:-0.0}
===========================================================
EOF
}

print_config

python "${PROJECT_DIR}/src/train_attention_cluster.py" \
  --config_dir "${CONFIG_DIR:-${REPO_ROOT}/fdong/Qwen3-0.6B}" \
  --output_dir "${OUT_DIR:-${PROJECT_DIR}/outputs/train}" \
  --run_name "${RUN_NAME:-moe-attention-cluster}" \
  --init_checkpoint "${INIT_CHECKPOINT:-}" \
  --total_steps "${TOTAL_STEPS:-10000}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --lr "${LR:-1e-3}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --warmup_steps "${WARMUP_STEPS:-100}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --save_interval "${SAVE_INTERVAL:-1000}" \
  --eval_interval "${EVAL_INTERVAL:-100}" \
  --eval_batches "${EVAL_BATCHES:-8}" \
  --log_interval "${LOG_INTERVAL:-10}" \
  --seed "${SEED:-1234}" \
  --device "${DEVICE:-cuda:0}" \
  --use_bf16 "${USE_BF16:-false}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-eager}" \
  --seq_len "${SEQ_LEN:-128}" \
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
  --moe_load_balance_loss_weight "${MOE_LOAD_BALANCE_LOSS_WEIGHT:-0.0}" \
  --gate_inhibition_weight "${GATE_INHIBITION_WEIGHT:-0.0}" \
  --gate_inhibition_temperature "${GATE_INHIBITION_TEMPERATURE:-1.0}" \
  --attention_cluster_weight "${ATTENTION_CLUSTER_WEIGHT:-0.05}" \
  --attention_cluster_temperature "${ATTENTION_CLUSTER_TEMPERATURE:-1.0}" \
  --attention_cluster_topk "${ATTENTION_CLUSTER_TOPK:-4}" \
  --attention_cluster_include_self "${ATTENTION_CLUSTER_INCLUDE_SELF:-false}" \
  --attention_cluster_detach_attention "${ATTENTION_CLUSTER_DETACH_ATTENTION:-true}" \
  --attention_cluster_negative_weight "${ATTENTION_CLUSTER_NEGATIVE_WEIGHT:-0.01}" \
  --attention_cluster_negative_feature_layer "${ATTENTION_CLUSTER_NEGATIVE_FEATURE_LAYER:-1}" \
  --attention_cluster_negative_history_only "${ATTENTION_CLUSTER_NEGATIVE_HISTORY_ONLY:-false}" \
  --expert_repulsion_weight "${EXPERT_REPULSION_WEIGHT:-0.0}" \
  --expert_repulsion_margin "${EXPERT_REPULSION_MARGIN:-0.0}" \
  "$@"
