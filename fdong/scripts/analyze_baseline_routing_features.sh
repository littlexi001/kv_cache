#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

RUNS="${RUNS:-inverse-kv-zipf-baseline}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
SYNTHETIC_SAMPLING_DISTRIBUTION="${SYNTHETIC_SAMPLING_DISTRIBUTION:-zipf}"  # uniform/zipf
OUTPUT_PATH="${OUTPUT_PATH:-../experiments/baseline_routing_feature_diagnostic_${SYNTHETIC_SAMPLING_DISTRIBUTION}_step${CHECKPOINT_STEP}.json}"

python3 analyze_baseline_routing_features.py \
  --runs "${RUNS}" \
  --checkpoint_step "${CHECKPOINT_STEP}" \
  --synthetic_sampling_distribution "${SYNTHETIC_SAMPLING_DISTRIBUTION}" \
  --attention_stride_pattern="1,1,1" \
  --residual_source_pattern="-1,-1,-1" \
  --output_path "${OUTPUT_PATH}"
