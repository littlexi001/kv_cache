#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong

BASE_OUT="ymluo/projects/learned_hierarchical_summary_memory/outputs"
SCRIPT="ymluo/projects/learned_hierarchical_summary_memory/src/run_blocksize_router_distill_from_sweeps.py"

DIRS=""
for block in 256 512 1024 2048; do
  for split in longbench ruler4k ruler8k ruler16k; do
    dir="$BASE_OUT/qwen8b_block${block}_topk_sweep_${split}_m3_20260707"
    if [[ -z "$DIRS" ]]; then
      DIRS="$dir"
    else
      DIRS="$DIRS,$dir"
    fi
  done
done

python "$SCRIPT" \
  --benchmark_output_dirs "$DIRS" \
  --output_dir "$BASE_OUT/blocksize_router_from_sweeps_m3_20260707" \
  --feature_block_tokens 512 \
  --hidden_dim 96 \
  --epochs 1200 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --test_fraction 0.30 \
  --summary_rouge_slack 0.03 \
  --quality_mode best_or_full \
  --seed 2026070731
