#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong

RUN_TAG="${RUN_TAG:-blocksize_floor_refinement_m20_20260708}"
MAX_EXAMPLES_PER_TASK="${MAX_EXAMPLES_PER_TASK:-20}"
GPU_LIST_CSV="${GPU_LIST_CSV:-0,1}"
SELECT_GROUPS_CSV="${SELECT_GROUPS_CSV:-ruler4k,ruler8k}"

METHODS="${METHODS:-full_raw,\
recent_plus_b128_span_top12_b0_a0,\
recent_plus_b128_span_top16_b0_a0,\
recent_plus_b256_span_top3_b0_a0,\
recent_plus_b256_span_top4_b0_a0,\
recent_plus_b256_span_top8_b0_a0,\
recent_plus_b256_span_top12_b0_a0,\
recent_plus_b512_span_top3_b0_a0,\
recent_plus_b512_span_top4_b0_a0,\
recent_plus_b512_span_top8_b0_a0}"

env \
  RUN_TAG="$RUN_TAG" \
  MAX_EXAMPLES_PER_TASK="$MAX_EXAMPLES_PER_TASK" \
  GPU_LIST_CSV="$GPU_LIST_CSV" \
  SELECT_GROUPS_CSV="$SELECT_GROUPS_CSV" \
  METHODS="$METHODS" \
  /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_blocksize_calibrated_heldout_20260708.sh
