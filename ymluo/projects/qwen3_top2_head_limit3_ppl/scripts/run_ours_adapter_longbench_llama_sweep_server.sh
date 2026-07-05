#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_ours_llama}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
SAMPLES="${SAMPLES:-1}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
MAX_NEW="${MAX_NEW:-64}"
PREFILL_CHUNK="${PREFILL_CHUNK:-2048}"
BUDGETS="${BUDGETS:-256 512 1024 2048}"
OURS_SCORER="${OURS_SCORER:-hybrid_late_mmr}"
METHODS="${METHODS:-full_kv,ours_page_gather}"

LONG_TASKS="${LONG_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,gov_report,multi_news}"

mkdir -p outputs/logs

for BUDGET in $BUDGETS; do
  OUT="outputs/controlled_public_kv_benchmark_ours_llama_longbench_${SAMPLES}shot_b${BUDGET}_${STAMP}"
  LOG="outputs/logs/controlled_public_kv_benchmark_ours_llama_longbench_${SAMPLES}shot_b${BUDGET}_${STAMP}.log"
  python src/run_controlled_public_kv_benchmark_v1.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --benchmarks longbench \
    --longbench_tasks "$LONG_TASKS" \
    --max_samples_per_task "$SAMPLES" \
    --max_context_tokens "$MAX_CONTEXT" \
    --max_new_tokens_override "$MAX_NEW" \
    --prefill_chunk_tokens "$PREFILL_CHUNK" \
    --methods "$METHODS" \
    --budget_tokens "$BUDGET" \
    --sink_tokens 64 \
    --recent_tokens 256 \
    --page_tokens 256 \
    --ours_scorer "$OURS_SCORER" \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    2>&1 | tee "$LOG"
done
