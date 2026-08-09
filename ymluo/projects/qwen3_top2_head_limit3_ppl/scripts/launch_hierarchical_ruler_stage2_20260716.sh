#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
LM_EVAL=/home/fdong/lm-evaluation-harness
TASKS=niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
PREFIX=20260716_hierarchical_ruler_64k128k_m5_shard
EXAMPLES=$ROOT/data/ruler_generated/llama31_8b_64k128k_m5_seed42.jsonl

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
cd "$ROOT"
mkdir -p outputs/logs

"$PYTHON" src/prepare_hierarchical_ruler_data_20260716.py \
  --model_name_or_path "$MODEL" \
  --lm_eval_path "$LM_EVAL" \
  --output "$EXAMPLES" \
  --ruler_tasks "$TASKS" \
  --ruler_lengths 65536,131072 \
  --max_samples_per_task 5 \
  --seed 42 \
  > outputs/logs/${PREFIX}_prepare.log 2>&1

for shard in 0 1; do
  if [[ "$shard" -eq 0 ]]; then
    devices=0,1,2,3
    cpus=0-23,48-71
  else
    devices=4,5,6,7
    cpus=24-47,72-95
  fi
  log="outputs/logs/${PREFIX}${shard}.log"
  CUDA_VISIBLE_DEVICES="$devices" taskset -c "$cpus" nohup "$PYTHON" \
    src/run_hierarchical_ruler_probe_20260716.py \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --examples_jsonl "$EXAMPLES" \
    --output_dir "outputs/${PREFIX}${shard}" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 65536,131072 \
    --max_samples_per_task 5 \
    --num_shards 2 \
    --shard_index "$shard" \
    --seed 42 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --projection_dim 64 \
    --index_bits 4 \
    --candidate_fraction 0.015 \
    --exact_cache_fraction 0.032 \
    --stream_group_size 2 \
    --hierarchical_prompt_mode full_prompt_then_compress \
    --prompt_wrapper none \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >> "$log" 2>&1 &
  echo "ruler-stage2 shard=$shard devices=$devices pid=$! log=$log"
done
