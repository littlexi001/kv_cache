#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
PREFIX=20260716_hierarchical_longbench_full_v1_shard

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH

cd "$ROOT"
mkdir -p outputs/logs

for shard in $(seq 0 7); do
  log="outputs/logs/${PREFIX}${shard}.log"
  CUDA_VISIBLE_DEVICES="$shard" nohup "$PYTHON" \
    src/run_hierarchical_longbench_probe_20260715.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "outputs/${PREFIX}${shard}" \
    --tasks "$TASKS" \
    --max_samples_per_task 0 \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.025 \
    --exact_cache_fraction 0.032 \
    --stream_group_size 1 \
    --prompt_wrapper llama3 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >> "$log" 2>&1 &
  echo "shard=$shard pid=$! log=$log"
done
