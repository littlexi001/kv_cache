#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_semantic_10k20k_smoke_v2_auth
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/analyze_longrange_book_index_semantic_retrieval.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir "$OUT" \
  --context_tokens 10000,20000 \
  --tasks_per_length 2 \
  --eval_tokens 64 \
  --chunk_size 256 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --top_fraction 0.02 \
  --exclude_sink_tokens 64 \
  --exclude_recent_tokens 512 \
  --fixed_page_size 64 \
  --paragraph_min_tokens 64 \
  --paragraph_max_tokens 192 \
  --section_max_paragraphs 8 \
  --query_window_tokens 256 \
  --flat_page_counts 4,8,16,32 \
  --hier_section_counts 1,2,4,8 \
  --hier_pages_per_section 2,4 \
  --observe_query_tokens last16 \
  --write_per_query false \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
