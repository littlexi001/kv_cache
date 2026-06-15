#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-inverse-kv-local-smoke}"
CONFIG_DIR="${CONFIG_DIR:-../configs/qwen3_0.6b}"
RUN_DIR="${RUN_DIR:-../runs/${RUN_NAME}}"

python3 single_thread_debug_qwen.py \
  --config_dir "$CONFIG_DIR" \
  --data_dir "unused-for-random-smoke" \
  --run_dir "$RUN_DIR" \
  --dataset_type random \
  --seq_len "${SEQ_LEN:-32}" \
  --local_batch_size "${LOCAL_BATCH_SIZE:-2}" \
  --global_batch_size "${GLOBAL_BATCH_SIZE:-2}" \
  --total_training_steps "${TOTAL_TRAINING_STEPS:-2}" \
  --save_interval "${SAVE_INTERVAL:-2}" \
  --warmup_steps 0 \
  --num_workers 0 \
  --use_bf16 false \
  --router_input "${ROUTER_INPUT:-k}" \
  --center_router_input "${CENTER_ROUTER_INPUT:-true}" \
  --num_experts "${NUM_EXPERTS:-4}" \
  --expert_intermediate_size "${EXPERT_INTERMEDIATE_SIZE:-32}" \
  --local_window "${LOCAL_WINDOW:-4}" \
  --sink_tokens "${SINK_TOKENS:-1}" \
  --debug_vocab_size "${DEBUG_VOCAB_SIZE:-256}" \
  --debug_hidden_size "${DEBUG_HIDDEN_SIZE:-64}" \
  --debug_num_hidden_layers "${DEBUG_NUM_HIDDEN_LAYERS:-2}" \
  --debug_num_attention_heads "${DEBUG_NUM_ATTENTION_HEADS:-4}" \
  --debug_num_key_value_heads "${DEBUG_NUM_KEY_VALUE_HEADS:-2}" \
  --debug_head_dim "${DEBUG_HEAD_DIM:-16}" \
  --debug_random_samples 16
