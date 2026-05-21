#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export SEQ_LEN="${SEQ_LEN:-128}"
export DATASET_TYPE="${DATASET_TYPE:-hierarchical_pattern}"  # jsonl/pruned/synthetic_indexed/hierarchical_pattern

export SYNTHETIC_NUM_SAMPLES="${SYNTHETIC_NUM_SAMPLES:-200000}"
export SYNTHETIC_BLOCK_SIZE="${SYNTHETIC_BLOCK_SIZE:-4}"
export SYNTHETIC_NUM_HIERARCHY_LAYERS="${SYNTHETIC_NUM_HIERARCHY_LAYERS:-2}"
export SYNTHETIC_CONTENT_TOKEN_COUNT="${SYNTHETIC_CONTENT_TOKEN_COUNT:-256}"
export SYNTHETIC_NUM_UNITS_PER_LAYER="${SYNTHETIC_NUM_UNITS_PER_LAYER:-64}"
export SYNTHETIC_SEED="${SYNTHETIC_SEED:-0}"
export SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.0}"
export SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

export DEBUG_VOCAB_SIZE="${DEBUG_VOCAB_SIZE:-257}"
export DEBUG_HIDDEN_SIZE="${DEBUG_HIDDEN_SIZE:-128}"
export DEBUG_INTERMEDIATE_SIZE="${DEBUG_INTERMEDIATE_SIZE:-256}"
export DEBUG_NUM_HIDDEN_LAYERS="${DEBUG_NUM_HIDDEN_LAYERS:-3}"
export DEBUG_NUM_ATTENTION_HEADS="${DEBUG_NUM_ATTENTION_HEADS:-4}"
export DEBUG_NUM_KEY_VALUE_HEADS="${DEBUG_NUM_KEY_VALUE_HEADS:-2}"
export DEBUG_HEAD_DIM="${DEBUG_HEAD_DIM:-32}"
export DEBUG_MAX_POSITION_EMBEDDINGS="${DEBUG_MAX_POSITION_EMBEDDINGS:-256}"

export USE_MOE="${USE_MOE:-true}"  # true/false
export MOE_NUM_UNIQUE_EXPERTS="${MOE_NUM_UNIQUE_EXPERTS:-4}"
export MOE_NUM_EXPERTS_PER_TOK="${MOE_NUM_EXPERTS_PER_TOK:-1}"
export MOE_INTERMEDIATE_SIZE="${MOE_INTERMEDIATE_SIZE:-128}"
export MOE_USE_COMMON_EXPERT="${MOE_USE_COMMON_EXPERT:-false}"  # true/false
export MOE_COMMON_INTERMEDIATE_SIZE="${MOE_COMMON_INTERMEDIATE_SIZE:--1}"
export MOE_ROUTER_TYPE="${MOE_ROUTER_TYPE:-linear}"  # linear/mlp
export MOE_ROUTER_HIDDEN_SIZE="${MOE_ROUTER_HIDDEN_SIZE:--1}"
export MOE_ROUTER_ACT="${MOE_ROUTER_ACT:-silu}"  # silu/relu/gelu/tanh
export MOE_HEAD_LEVEL="${MOE_HEAD_LEVEL:-false}"  # true/false
export MOE_LOAD_BALANCE_LOSS_WEIGHT="${MOE_LOAD_BALANCE_LOSS_WEIGHT:-0.0}"
export MOE_ROUTER_INHIBITION_LOSS_WEIGHT="${MOE_ROUTER_INHIBITION_LOSS_WEIGHT:-0.05}"
export MOE_ROUTER_INHIBITION_TEMPERATURE="${MOE_ROUTER_INHIBITION_TEMPERATURE:-1.0}"

export GROUND_TRUTH_ROUTING_MODE="${GROUND_TRUTH_ROUTING_MODE:-dispatch}"  # dispatch/supervise
export GROUND_TRUTH_ROUTING_STRATEGY="${GROUND_TRUTH_ROUTING_STRATEGY:-none}"  # none/hash/frequency_balanced
export GROUND_TRUTH_ROUTING_FEATURE_LAYER="${GROUND_TRUTH_ROUTING_FEATURE_LAYER:-1}"  # 0=local slot, 1=higher-level unit
export GROUND_TRUTH_FREQUENCY_ESTIMATE_SAMPLES="${GROUND_TRUTH_FREQUENCY_ESTIMATE_SAMPLES:-4096}"

export ATTENTION_STRIDE_PATTERN="${ATTENTION_STRIDE_PATTERN:-1,1,1}"
export RESIDUAL_SOURCE_PATTERN="${RESIDUAL_SOURCE_PATTERN:--1,-1,-1}"
export LR="${LR:-1e-3}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5000}"
export TRAINING_SEED="${TRAINING_SEED:-0}"

mkdir -p ../logs ../experiments ../checkpoints

run_one() {
  local distribution="$1"
  local router_input="$2"  # hidden/attention_output
  local run_name="inhibition-${distribution}-${router_input}"

  echo "==== training ${run_name} ===="
  SYNTHETIC_SAMPLING_DISTRIBUTION="${distribution}" \
  MOE_ROUTER_INPUT="${router_input}" \
  CKPT_DIR="../checkpoints/${run_name}" \
  bash single_thread_debug.sh > "../logs/${run_name}.log" 2>&1
}

run_one uniform hidden
run_one uniform attention_output
run_one zipf hidden
run_one zipf attention_output

python3 analyze_moe_variant_selectivity.py \
  --checkpoint_root ../checkpoints \
  --runs "inhibition-uniform-hidden,inhibition-uniform-attention_output" \
  --checkpoint_step "${TOTAL_TRAINING_STEPS}" \
  --seq_len "${SEQ_LEN}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
  --synthetic_seed "${SYNTHETIC_SEED}" \
  --synthetic_sampling_distribution uniform \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE}" \
  --debug_intermediate_size "${DEBUG_INTERMEDIATE_SIZE}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS}" \
  --debug_head_dim "${DEBUG_HEAD_DIM}" \
  --debug_max_position_embeddings "${DEBUG_MAX_POSITION_EMBEDDINGS}" \
  --attention_stride_pattern="${ATTENTION_STRIDE_PATTERN}" \
  --residual_source_pattern="${RESIDUAL_SOURCE_PATTERN}" \
  --output_path "../experiments/inhibition_selectivity_uniform_step${TOTAL_TRAINING_STEPS}.json"

python3 analyze_moe_variant_selectivity.py \
  --checkpoint_root ../checkpoints \
  --runs "inhibition-zipf-hidden,inhibition-zipf-attention_output" \
  --checkpoint_step "${TOTAL_TRAINING_STEPS}" \
  --seq_len "${SEQ_LEN}" \
  --synthetic_block_size "${SYNTHETIC_BLOCK_SIZE}" \
  --synthetic_num_hierarchy_layers "${SYNTHETIC_NUM_HIERARCHY_LAYERS}" \
  --synthetic_content_token_count "${SYNTHETIC_CONTENT_TOKEN_COUNT}" \
  --synthetic_num_units_per_layer "${SYNTHETIC_NUM_UNITS_PER_LAYER}" \
  --synthetic_seed "${SYNTHETIC_SEED}" \
  --synthetic_sampling_distribution zipf \
  --synthetic_zipf_alpha "${SYNTHETIC_ZIPF_ALPHA}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE}" \
  --debug_intermediate_size "${DEBUG_INTERMEDIATE_SIZE}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS}" \
  --debug_head_dim "${DEBUG_HEAD_DIM}" \
  --debug_max_position_embeddings "${DEBUG_MAX_POSITION_EMBEDDINGS}" \
  --attention_stride_pattern="${ATTENTION_STRIDE_PATTERN}" \
  --residual_source_pattern="${RESIDUAL_SOURCE_PATTERN}" \
  --output_path "../experiments/inhibition_selectivity_zipf_step${TOTAL_TRAINING_STEPS}.json"

echo "==== done ===="
echo "logs: ../logs/inhibition-*.log"
echo "checkpoints: ../checkpoints/inhibition-*"
echo "analysis: ../experiments/inhibition_selectivity_{uniform,zipf}_step${TOTAL_TRAINING_STEPS}.json"
