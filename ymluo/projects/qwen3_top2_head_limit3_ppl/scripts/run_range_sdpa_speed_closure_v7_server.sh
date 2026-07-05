#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_smoke}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
PY="${PY:-python}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-10000,20000}"
TASKS_PER_LENGTH="${TASKS_PER_LENGTH:-2}"
EVAL_TOKENS="${EVAL_TOKENS:-32}"
MODES="${MODES:-full,chain_typedhier_role_auto_p1}"
LAYOUTS="${LAYOUTS:-e05_d90,e20_d80}"

OUT="outputs/range_sdpa_speed_closure_v7_${STAMP}"
LOG="outputs/logs/range_sdpa_speed_closure_v7_${STAMP}.log"
mkdir -p outputs/logs "$OUT"

"$PY" src/run_longrange_book_index_sparse_eval.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --context_tokens "$CONTEXT_TOKENS" \
  --tasks_per_length "$TASKS_PER_LENGTH" \
  --eval_tokens "$EVAL_TOKENS" \
  --task_variant chain_story_conflict \
  --suite_layouts "$LAYOUTS" \
  --modes "$MODES" \
  --score_query_ppl true \
  --score_calibrated true \
  --balanced_labels true \
  --answer_score_format gated_sentence \
  --sparse_attention_impl range_sdpa \
  --typed_record_mode extractive \
  --typed_record_format answerline_summary \
  --typed_summary_source_mode chain_typedhier_auto_p1 \
  --typed_record_answer_override true \
  --skip_lm_answer_when_override true \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  --attn_implementation eager \
  2>&1 | tee "$LOG"

echo "$OUT"
