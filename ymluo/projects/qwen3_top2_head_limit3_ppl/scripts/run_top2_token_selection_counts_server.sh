#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_token_selection_counts_war_4k_v1
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/analyze_top2_token_selection_counts.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --text_path data/war_and_peace_pg2600.txt \
  --output_dir "$OUT" \
  --total_tokens 4160 \
  --prefill_tokens 4096 \
  --eval_tokens 64 \
  --chunk_size 64 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --top_fraction 0.02 \
  --max_query_samples 64 \
  --include_token_text true \
  --write_zero_count_tokens true \
  --write_zero_count_layer_rows false \
  --write_layer_head_token_counts true \
  --layer_head_min_count 1 \
  --top_tokens_per_layer_head 100 \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
