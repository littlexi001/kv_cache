#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../../.." && pwd)"

export TOKENIZERS_PARALLELISM=false

python "${PROJECT_DIR}/src/train_rule_permutation.py" \
  --config_dir "${CONFIG_DIR:-${REPO_ROOT}/fdong/Qwen3-0.6B}" \
  --output_dir "${OUT_DIR:-${PROJECT_DIR}/outputs/train}" \
  --run_name "${RUN_NAME:-moe-rule-permutation}" \
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
  --rule_num_samples "${RULE_NUM_SAMPLES:-200000}" \
  --rule_block_len "${RULE_BLOCK_LEN:-16}" \
  --rule_vocab_size "${RULE_VOCAB_SIZE:-256}" \
  --rule_num_high_units "${RULE_NUM_HIGH_UNITS:-64}" \
  --rule_num_groups "${RULE_NUM_GROUPS:-4}" \
  --rule_num_rules_per_group "${RULE_NUM_RULES_PER_GROUP:-1}" \
  --rule_seed "${RULE_SEED:-0}" \
  --rule_pad_token_id "${RULE_PAD_TOKEN_ID:-0}" \
  --rule_min_token_id "${RULE_MIN_TOKEN_ID:-1}" \
  --rule_slot_assignment "${RULE_SLOT_ASSIGNMENT:-mod}" \
  --rule_per_occurrence "${RULE_PER_OCCURRENCE:-slot_fixed}" \
  --rule_start_mode "${RULE_START_MODE:-slot_fixed}" \
  --loss_rule_positions_only "${LOSS_RULE_POSITIONS_ONLY:-false}" \
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
  --moe_intermediate_size "${MOE_INTERMEDIATE_SIZE:-64}" \
  --moe_use_common_expert "${MOE_USE_COMMON_EXPERT:-false}" \
  --moe_common_intermediate_size "${MOE_COMMON_INTERMEDIATE_SIZE:--1}" \
  --moe_router_bias "${MOE_ROUTER_BIAS:-false}" \
  --moe_normalize_topk_prob "${MOE_NORMALIZE_TOPK_PROB:-true}" \
  --moe_router_input "${MOE_ROUTER_INPUT:-attention_output}" \
  --moe_head_level "${MOE_HEAD_LEVEL:-false}" \
  --gate_inhibition_weight "${GATE_INHIBITION_WEIGHT:-0.05}" \
  --gate_inhibition_temperature "${GATE_INHIBITION_TEMPERATURE:-1.0}" \
  --expert_repulsion_weight "${EXPERT_REPULSION_WEIGHT:-0.0}" \
  --expert_repulsion_margin "${EXPERT_REPULSION_MARGIN:-0.0}" \
  --orthogonalize_gate "${ORTHOGONALIZE_GATE:-false}" \
  --orthogonalize_experts "${ORTHOGONALIZE_EXPERTS:-false}" \
  --orthogonal_init_mode "${ORTHOGONAL_INIT_MODE:-preserve_norm}" \
  --orthogonalize_after_checkpoint "${ORTHOGONALIZE_AFTER_CHECKPOINT:-false}" \
  --routing_override "${ROUTING_OVERRIDE:-none}" \
  --all_to_one_expert "${ALL_TO_ONE_EXPERT:-0}" \
  --eval_oracle_routing "${EVAL_ORACLE_ROUTING:-true}" \
  --eval_all_to_one_routing "${EVAL_ALL_TO_ONE_ROUTING:-true}" \
  "$@"
