#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
OUTPUT=$ROOT/results/20260729_qksieve_low192_unbiased_cuda/multiseed

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export CUDA_VISIBLE_DEVICES=5
export TORCH_CUDA_ARCH_LIST=8.6

mkdir -p "$OUTPUT"
cd "$ROOT"

run_one() {
  local tag=$1
  local profile=$2
  local seed=$3
  shift 3
  "$PYTHON" -u src/benchmark_variablebit_spectral_attention_20260727.py \
    --lengths 32768,65536,131072 \
    --allocation_profile "$profile" \
    --adaptive_sample_count \
    --seed "$seed" \
    --warmup 10 \
    --iterations 60 \
    --full_iterations 20 \
    --split_count 16 \
    --output "$OUTPUT/${tag}_${seed}.json" \
    "$@" \
    >/dev/null
}

for seed in 11 23 37 53 71; do
  run_one qksieve qmse_total_b15 "$seed"
  run_one low192_unbiased fixed_low192 "$seed" --unbiased_quantile
done

touch "$OUTPUT/ALL_COMPLETE"
