#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
SAMPLES="${SAMPLES:-3}"
BUDGET="${BUDGET:-512}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
MAX_NEW="${MAX_NEW:-32}"
OUT="outputs/controlled_public_kv_benchmark_v1_lb4_ruler_${SAMPLES}shot_b${BUDGET}_${STAMP}"
LOG="outputs/logs/controlled_public_kv_benchmark_v1_lb4_ruler_${SAMPLES}shot_b${BUDGET}_${STAMP}.log"

mkdir -p outputs/logs

python src/run_controlled_public_kv_benchmark_v1.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --benchmarks longbench,ruler \
  --longbench_tasks passage_retrieval_en,hotpotqa,2wikimqa,multifieldqa_en \
  --ruler_tasks niah_single_1 \
  --max_samples_per_task "$SAMPLES" \
  --max_context_tokens "$MAX_CONTEXT" \
  --max_new_tokens_override "$MAX_NEW" \
  --methods full_kv,streamingllm_sink_recent,h2o_observe,snapkv_observe,ours_page_gather \
  --budget_tokens "$BUDGET" \
  --sink_tokens 64 \
  --recent_tokens 256 \
  --page_tokens 256 \
  --ruler_lengths 4096 \
  --dtype float16 \
  --attn_implementation eager \
  2>&1 | tee "$LOG"
