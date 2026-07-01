#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m py_compile src/book_page_router.py src/run_longrange_book_index_sparse_eval.py

OUT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v13_sdpa_gqa_gather_smoke"
mkdir -p "$OUT"

python -u src/run_longrange_book_index_sparse_eval.py \
  --output_dir "$OUT" \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --context_tokens 20000 \
  --tasks_per_length 1 \
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
  --suite_layouts e05_d90 \
  --modes sink_recent,remote_tail_p4,book_auth_flat_p4,budget_authflat_p4_authadj2_b4 \
  --score_query_ppl true \
  --score_calibrated true \
  --balanced_labels true \
  --answer_score_format answer_label \
  --sparse_attention_impl sdpa_gather \
  > "$OUT/run.log" 2>&1

cat "$OUT/run.log"
column -s, -t "$OUT/sparse_summary.csv"
