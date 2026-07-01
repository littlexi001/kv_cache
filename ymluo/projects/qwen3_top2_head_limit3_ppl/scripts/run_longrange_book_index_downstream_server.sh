#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_downstream_10k20k_smoke_v2_calib
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup python -u src/run_longrange_book_index_downstream_eval.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir "$OUT" \
  --context_tokens 10000,20000 \
  --tasks_per_length 2 \
  --eval_tokens 64 \
  --chunk_size 512 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  --sink_tokens 64 \
  --recent_tokens 512 \
  --paragraph_min_tokens 64 \
  --paragraph_max_tokens 192 \
  --section_max_paragraphs 8 \
  --query_window_tokens 256 \
  --schemes full_context,recent_only,sink_recent,remote_tail_p4,remote_tail_p8,remote_tail_p16,book_flat_p4,book_flat_p8,book_auth_flat_p4,book_auth_flat_p8,book_auth_flat_p16,book_hier_s4_p2,book_auth_hier_s4_p2,hybrid_tail4_authflat4,hybrid_tail4_authhier_s4_p2 \
  --add_page_markers true \
  > "$OUT/run.log" 2>&1 < /dev/null &

echo "$!" > "$OUT/pid.txt"
echo "started $(cat "$OUT/pid.txt")"
echo "log $OUT/run.log"
