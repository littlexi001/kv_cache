#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hierarchical_book_index_war_4k_s64_r512_v3_tailctrl
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/analyze_hierarchical_book_index_recall.py \
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
  --exclude_sink_tokens 64 \
  --exclude_recent_tokens 512 \
  --fixed_page_size 64 \
  --min_sentence_tokens 8 \
  --paragraph_min_tokens 64 \
  --paragraph_max_tokens 192 \
  --section_max_paragraphs 8 \
  --query_window_tokens 256 \
  --flat_page_counts 4,8,16 \
  --hier_section_counts 1,2,4 \
  --hier_pages_per_section 2,4 \
  --max_query_samples 64 \
  --write_per_query true \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
