#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROJECT=/home/fdong/ymluo/projects/qwen3_k_svd_conflict_geometry
MODEL=/home/fdong/hrj/prove/Qwen3-0.6B
OUTPUT="$PROJECT/outputs/k_svd_geometry_64seed_final_20260717"

mkdir -p "$OUTPUT"
for gpu in $(seq 0 7); do
  seed_start=$((gpu * 8))
  shard=$(printf "shard_%02d" "$gpu")
  mkdir -p "$OUTPUT/$shard"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" \
    "$PROJECT/src/run_k_svd_conflict_geometry.py" \
    --model "$MODEL" \
    --output-dir "$OUTPUT/$shard" \
    --seed-start "$seed_start" \
    --num-seeds 8 \
    --pair-contexts short,filler_8k \
    --ranks 4,8,16,32,64 \
    --filler-tokens 8192 \
    --chain-length 2 \
    --prefill-chunk 1024 \
    > "$OUTPUT/$shard/run.log" 2>&1 &
  echo "$!" > "$OUTPUT/$shard/pid"
  echo "started gpu=$gpu seed_start=$seed_start pid=$!"
done
