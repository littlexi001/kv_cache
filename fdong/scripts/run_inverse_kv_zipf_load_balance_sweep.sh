#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

mkdir -p ../logs ../checkpoints

export SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"
export SYNTHETIC_ZIPF_ALPHA="${SYNTHETIC_ZIPF_ALPHA:-1.1}"
export SYNTHETIC_ZIPF_SHUFFLE_RANKS="${SYNTHETIC_ZIPF_SHUFFLE_RANKS:-true}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

export MOE_ROUTER_INPUT="${MOE_ROUTER_INPUT:-attention_output}"
export MOE_HEAD_LEVEL="${MOE_HEAD_LEVEL:-false}"

RUN_PREFIX="${RUN_PREFIX:-inverse-kv-zipf-attn-output-router-lb}"
WEIGHTS="${WEIGHTS:-0.001 0.01 0.1}"

for weight in $WEIGHTS; do
  safe_weight="${weight//./p}"
  safe_weight="${safe_weight//-/_}"
  export MOE_LOAD_BALANCE_LOSS_WEIGHT="$weight"
  export RUN_NAME="${RUN_PREFIX}-${safe_weight}"
  export CKPT_DIR="../checkpoints/${RUN_NAME}"

  echo "[$(date)] start ${RUN_NAME} weight=${weight}"
  bash run_inverse_kv_local_experiment.sh > "../logs/${RUN_NAME}.train.log" 2>&1
  echo "[$(date)] done ${RUN_NAME}"
done

echo "[$(date)] load-balance sweep done"
