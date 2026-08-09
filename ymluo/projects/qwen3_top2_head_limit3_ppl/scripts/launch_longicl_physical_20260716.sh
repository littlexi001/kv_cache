#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=$ROOT/datasets/lbv2_frozen_splits_20260714/router_calibration.json
PREFIX=20260716_longicl_physical_calibration_m14_shard

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p outputs/logs

for shard in $(seq 0 7); do
  if [[ "$shard" -lt 4 ]]; then cpus=0-23,48-71; else cpus=24-47,72-95; fi
  CUDA_VISIBLE_DEVICES="$shard" taskset -c "$cpus" nohup "$PYTHON" \
    src/run_hierarchical_longicl_probe_20260716.py \
    --model_name_or_path "$MODEL" \
    --longbench_v2_json "$DATA" \
    --output_dir "outputs/${PREFIX}${shard}" \
    --max_samples 0 \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_context_tokens 32000 \
    --max_new_tokens 32 \
    --prefill_chunk_tokens 2048 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --exact_cache_fraction 0.032 \
    --stream_group_size 2 \
    --hierarchical_prompt_mode full_prompt_then_compress \
    --prompt_wrapper llama3 \
    --dtype float16 --device cuda --device_map auto \
    > "outputs/logs/${PREFIX}${shard}.log" 2>&1 &
  echo "longicl shard=$shard pid=$!"
done
