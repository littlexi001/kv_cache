#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints

export SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"  # uniform/zipf
export SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.1}"
export SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"  # true/false

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"

RUN_PREFIX="${RUN_PREFIX:-inverse-kv-zipf}"

run_variant() {
  local suffix="$1"
  local router_input="$2"
  local head_level="$3"

  export RUN_NAME="${RUN_PREFIX}-${suffix}"
  export CKPT_DIR="../checkpoints/${RUN_NAME}"
  export MOE_ROUTER_INPUT="${router_input}"  # hidden/attention_output
  export MOE_HEAD_LEVEL="${head_level}"  # true/false

  echo "[$(date)] start ${RUN_NAME}"
  bash run_inverse_kv_local_experiment.sh > "../logs/${RUN_NAME}.train.log" 2>&1
  echo "[$(date)] done ${RUN_NAME}"
}

run_variant "baseline" "hidden" "false"
run_variant "attn-output-router" "attention_output" "false"
run_variant "head-moe-hidden-router" "hidden" "true"
run_variant "attn-output-head-moe" "attention_output" "true"

echo "[$(date)] all zipf variants done"
