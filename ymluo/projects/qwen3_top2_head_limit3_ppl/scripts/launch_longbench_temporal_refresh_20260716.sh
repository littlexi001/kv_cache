#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
TASKS=narrativeqa,hotpotqa,passage_retrieval_en,lcc

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p outputs/logs

for interval in 2 4; do
  for shard in 0 1 2 3; do
    if [[ "$interval" -eq 2 ]]; then
      device=$shard
    else
      device=$((shard + 4))
    fi
    if [[ "$device" -lt 4 ]]; then cpus=0-23,48-71; else cpus=24-47,72-95; fi
    prefix="20260716_longbench_temporal_refresh_i${interval}_m20_shard"
    log="outputs/logs/${prefix}${shard}.log"
    CUDA_VISIBLE_DEVICES="$device" taskset -c "$cpus" nohup "$PYTHON" \
      src/run_hierarchical_longbench_probe_20260715.py \
      --model_name_or_path "$MODEL" \
      --longbench_data_dir "$DATA" \
      --output_dir "outputs/${prefix}${shard}" \
      --tasks "$TASKS" \
      --max_samples_per_task 20 \
      --num_shards 4 \
      --shard_index "$shard" \
      --max_context_tokens 7500 \
      --max_new_tokens_override 0 \
      --prefill_chunk_tokens 2048 \
      --projection_dim 64 \
      --index_bits 4 \
      --candidate_fraction 0.025 \
      --exact_cache_fraction 0.032 \
      --stream_group_size 1 \
      --candidate_refresh_interval "$interval" \
      --hierarchical_prompt_mode full_prompt_then_compress \
      --prefill_cache_mode dynamic \
      --prompt_wrapper llama3 \
      --dtype float16 \
      --device cuda \
      --device_map auto \
      >> "$log" 2>&1 &
    echo "temporal interval=$interval shard=$shard device=$device pid=$! log=$log"
  done
done
