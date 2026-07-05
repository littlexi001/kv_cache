#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_sparseprefill}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
PREDICTOR_PATH="${PREDICTOR_PATH:-outputs/causal_page_sparse_prefill_v6_1shot_b512_${STAMP}/causal_page_predictor.json}"
TEXT_CASES="${TEXT_CASES:-war:data/war_and_peace_pg2600.txt,monte:data/count_monte_cristo_pg1184.txt}"
PREFILL_TOKENS="${PREFILL_TOKENS:-8192}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
BUDGET="${BUDGET:-512}"
OUT="outputs/causal_page_sparse_prefill_ppl_v6_p${PREFILL_TOKENS}_e${EVAL_TOKENS}_b${BUDGET}_${STAMP}"
LOG="outputs/logs/causal_page_sparse_prefill_ppl_v6_p${PREFILL_TOKENS}_e${EVAL_TOKENS}_b${BUDGET}_${STAMP}.log"

mkdir -p outputs/logs

python src/run_causal_page_sparse_prefill_ppl_v6.py \
  --model_name_or_path "$MODEL" \
  --predictor_path "$PREDICTOR_PATH" \
  --output_dir "$OUT" \
  --text_cases "$TEXT_CASES" \
  --prefill_tokens "$PREFILL_TOKENS" \
  --eval_tokens "$EVAL_TOKENS" \
  --eval_chunk_size 64 \
  --budget_tokens "$BUDGET" \
  --sink_tokens 64 \
  --recent_tokens 256 \
  --page_tokens 256 \
  --dtype float16 \
  --attn_implementation sdpa \
  2>&1 | tee "$LOG"
