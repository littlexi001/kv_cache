#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

BASE_OUT="ymluo/projects/learned_hierarchical_summary_memory/outputs"
SCRIPT="ymluo/projects/learned_hierarchical_summary_memory/src/run_blocksize_floor_calibration_from_sweeps.py"
RUN_TAG="${RUN_TAG:-blocksize_floor_calibration_m3_20260708}"
ALPHA="${ALPHA:-0.0}"

DIRS=""
for block in 32 64 128 256 512; do
  for split in longbench ruler4k ruler8k ruler16k; do
    dir="$BASE_OUT/qwen8b_block${block}_topk_sweep_${split}_m3_20260707"
    if [[ -z "$DIRS" ]]; then
      DIRS="$dir"
    else
      DIRS="$DIRS,$dir"
    fi
  done
done

ALLOW='recent_plus_b(32_span_top(1|2|4|8|16|32)_b0_a0|64_span_top(1|2|4|8|16)_b0_a0|128_span_top(2|3|6|12)_b0_a0|256_span_top(2|3|4|8)_b0_a0|512_span_top(2|3|4)_b0_a0)'

python "$SCRIPT" \
  --benchmark_output_dirs "$DIRS" \
  --output_dir "$BASE_OUT/$RUN_TAG" \
  --feature_block_tokens 512 \
  --summary_rouge_slack 0.03 \
  --quality_mode best_or_full \
  --allowed_label_regex "$ALLOW" \
  --alpha "$ALPHA" \
  --min_cases 3
