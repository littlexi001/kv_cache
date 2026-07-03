#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
TASKS="${TASKS:-qasper,hotpotqa,passage_retrieval_en}"
SAMPLES="${SAMPLES:-1}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
MAX_NEW="${MAX_NEW:-32}"
TARGET_TOKENS="${TARGET_TOKENS:-16}"
MAX_LABEL_PAGES="${MAX_LABEL_PAGES:-10}"
BUDGET="${BUDGET:-512}"
OUT="outputs/causal_page_influence_predictor_v6_${SAMPLES}shot_b${BUDGET}_${STAMP}"
LOG="outputs/logs/causal_page_influence_predictor_v6_${SAMPLES}shot_b${BUDGET}_${STAMP}.log"

mkdir -p outputs/logs

python src/run_causal_page_influence_predictor_v6.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --longbench_tasks "$TASKS" \
  --max_samples_per_task "$SAMPLES" \
  --max_context_tokens "$MAX_CONTEXT" \
  --max_new_tokens_override "$MAX_NEW" \
  --target_tokens "$TARGET_TOKENS" \
  --max_label_pages "$MAX_LABEL_PAGES" \
  --budget_tokens "$BUDGET" \
  --sink_tokens 64 \
  --recent_tokens 256 \
  --page_tokens 256 \
  --dtype float16 \
  --attn_implementation sdpa \
  --prompt_wrapper llama3 \
  2>&1 | tee "$LOG"
