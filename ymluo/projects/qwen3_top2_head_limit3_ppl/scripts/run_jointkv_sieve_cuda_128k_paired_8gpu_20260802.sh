#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RUNNER="$ROOT/src/benchmark_jointkv_sieve_direct_stages_20260802.py"
OUTPUT="$ROOT/results/20260802_jointkv_sieve_cuda_128k_paired_8gpu"
mkdir -p "$OUTPUT"
export PYTHONPATH="$ROOT/src"
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export TORCH_EXTENSIONS_DIR=/home/fdong/.cache/torch_extensions_jointkv
export TORCH_CUDA_ARCH_LIST=8.6

run_pair() {
  local gpu=$1
  for bits in 64 48; do
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "$PYTHON" "$RUNNER" \
        --length 131072 --base_bits "$bits" --residual_bits 48 \
        --refine_fraction 0.20 --warmup 12 --iterations 50 \
        --seed "$((20260820 + gpu))" \
        --output "$OUTPUT/gpu${gpu}_b${bits}.json" \
        >"$OUTPUT/gpu${gpu}_b${bits}.log" 2>&1
  done
}

for gpu in 0 1 2 3 4 5 6 7; do
  run_pair "$gpu" &
done
wait
touch "$OUTPUT/ALL_COMPLETE"
