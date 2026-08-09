#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RUNNER="$ROOT/src/benchmark_jointkv_sieve_direct_stages_20260802.py"
OUTPUT="$ROOT/results/20260802_jointkv_sieve_cuda_matrix_8gpu"
mkdir -p "$OUTPUT"
export PYTHONPATH="$ROOT/src"
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export TORCH_EXTENSIONS_DIR=/home/fdong/.cache/torch_extensions_jointkv
export TORCH_CUDA_ARCH_LIST=8.6

run_one() {
  local gpu=$1 length=$2 bits=$3
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$RUNNER" \
      --length "$length" --base_bits "$bits" --residual_bits 48 \
      --refine_fraction 0.20 --warmup 8 --iterations 20 \
      --output "$OUTPUT/jointkv_b${bits}_n${length}.json" \
      >"$OUTPUT/jointkv_b${bits}_n${length}.log" 2>&1
}

run_one 0 8192 64 &
run_one 1 16384 64 &
run_one 2 32768 64 &
run_one 3 65536 64 &
run_one 4 131072 64 &
run_one 5 32768 48 &
run_one 6 65536 48 &
run_one 7 131072 48 &
wait

run_one 0 8192 48 &
run_one 1 16384 48 &
wait
touch "$OUTPUT/ALL_COMPLETE"
