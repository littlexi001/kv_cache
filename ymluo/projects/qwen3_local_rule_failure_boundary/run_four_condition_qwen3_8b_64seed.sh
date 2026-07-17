#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUTPUT="$PROJECT/outputs/four_condition_qwen3_8b_64seed_20260717"

mkdir -p "$OUTPUT"
for gpu in $(seq 0 7); do
  seed_start=$((gpu * 8))
  shard=$(printf "shard_%02d" "$gpu")
  mkdir -p "$OUTPUT/$shard"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" \
    "$PROJECT/src/run_four_condition_answer_eval_20260717.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT/$shard" \
    --seed_start "$seed_start" \
    --num_seeds 8 \
    --filler_tokens 8192 \
    --chain_length 2 \
    --candidate_count 8 \
    --max_new_tokens 256 \
    --report_generation_budgets 16,128,256 \
    --prefill_chunk_size 1024 \
    --dtype float16 \
    --device_map auto \
    --attn_implementation sdpa \
    > "$OUTPUT/$shard/run.log" 2>&1 &
  echo "$!" > "$OUTPUT/$shard/pid"
  echo "started gpu=$gpu seed_start=$seed_start pid=$!"
  sleep 3
done
