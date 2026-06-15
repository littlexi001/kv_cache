#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

RUN_NAME="${RUN_NAME:-inverse-kv-qwen3-0.6b-k-centered-e4}"
CONFIG_DIR="${CONFIG_DIR:-../../../Qwen3-0.6B}"
# Keep this server path paired with TokenizedJSONLData.
DATA_DIR="${DATA_DIR:-../../../dclm/global-shard_01_of_10}"
RUN_DIR="${RUN_DIR:-../runs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-../logs}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-12345}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
SEQ_LEN="${SEQ_LEN:-1024}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"

ARCHITECTURE="${ARCHITECTURE:-shared_bucket}"          # ordinary_moe | shared_bucket
ROUTER_INPUT="${ROUTER_INPUT:-k}"                     # layer_input | q | k | v
CENTER_ROUTER_INPUT="${CENTER_ROUTER_INPUT:-true}"    # true | false
ROUTER_NORMALIZATION="${ROUTER_NORMALIZATION:-none}"    # l2 | none
NUM_EXPERTS="${NUM_EXPERTS:-4}"
EXPERT_INTERMEDIATE_SIZE="${EXPERT_INTERMEDIATE_SIZE:-3072}"
LOCAL_WINDOW="${LOCAL_WINDOW:-32}"
SINK_TOKENS="${SINK_TOKENS:-4}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

nohup torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_addr=localhost \
  --master_port="$MASTER_PORT" \
  pretrain_qwen.py \
  --config_dir "$CONFIG_DIR" \
  --data_dir "$DATA_DIR" \
  --run_dir "$RUN_DIR" \
  --dataset_type dclm \
  --local_batch_size "$LOCAL_BATCH_SIZE" \
  --global_batch_size "$GLOBAL_BATCH_SIZE" \
  --seq_len "$SEQ_LEN" \
  --total_training_steps "$TOTAL_TRAINING_STEPS" \
  --save_interval "$SAVE_INTERVAL" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_steps "$WARMUP_STEPS" \
  --architecture "$ARCHITECTURE" \
  --router_input "$ROUTER_INPUT" \
  --center_router_input "$CENTER_ROUTER_INPUT" \
  --router_normalization "$ROUTER_NORMALIZATION" \
  --num_experts "$NUM_EXPERTS" \
  --expert_intermediate_size "$EXPERT_INTERMEDIATE_SIZE" \
  --local_window "$LOCAL_WINDOW" \
  --sink_tokens "$SINK_TOKENS" \
  >"$LOG_DIR/${RUN_NAME}.log" 2>&1 &

echo "Started PID $!"
echo "Console log: $LOG_DIR/${RUN_NAME}.log"
echo "Metrics:     $RUN_DIR/train_metrics.jsonl"
echo "Monitor:     tail -f $LOG_DIR/${RUN_NAME}.log"
