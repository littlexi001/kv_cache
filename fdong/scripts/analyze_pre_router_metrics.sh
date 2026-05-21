#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

RUNS="${RUNS:-pre-router-zipf-layer_input-kl,pre-router-zipf-layer_input-pairwise,pre-router-zipf-layer_input-topk_logits}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"  # uniform/zipf
PRE_ROUTER_INPUT="${PRE_ROUTER_INPUT:-layer_input}"  # layer_input/q
RHO="${RHO:-0.75}"
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/pre_router_metrics_${SYNTHETIC_SAMPLING_DISTRIBUTION}_${PRE_ROUTER_INPUT}_step${CHECKPOINT_STEP}.json}"

python3 analyze_pre_router_metrics.py \
  --runs "${RUNS}" \
  --checkpoint_step "${CHECKPOINT_STEP}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION}" \
  --rho "${RHO}" \
  --attention_stride_pattern="1,1,1" \
  --residual_source_pattern="-1,-1,-1" \
  --output_path "${OUTPUT_PATH}"
