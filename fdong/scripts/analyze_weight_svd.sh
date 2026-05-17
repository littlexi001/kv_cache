#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-../checkpoints}"
RUNS="${RUNS:-inverse-kv-local-h128-l3-top1,inverse-kv-attn-output-router,inverse-kv-head-moe-hidden-router,inverse-kv-attn-output-head-moe}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-../experiments/weight_svd_step${CHECKPOINT_STEP}}"
INCLUDE_ROUTER="${INCLUDE_ROUTER:-true}"  # true/false
PLOT="${PLOT:-true}"  # true/false

ARGS=""
ARGS+=" --checkpoint_root $CHECKPOINT_ROOT"
ARGS+=" --runs $RUNS"
ARGS+=" --checkpoint_step $CHECKPOINT_STEP"
ARGS+=" --output_dir $OUTPUT_DIR"

if [ "$INCLUDE_ROUTER" = "true" ]; then
  ARGS+=" --include_router"
else
  ARGS+=" --no_include_router"
fi

if [ "$PLOT" = "true" ]; then
  ARGS+=" --plot"
else
  ARGS+=" --no_plot"
fi

python3 analyze_weight_svd.py ${ARGS}
