#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="${STAMP:-20260703_v2}"
MODEL="${MODEL:-/home/fdong/hrj/prove/Qwen3-0.6B}"
SAMPLES="${SAMPLES:-1}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
MAX_NEW="${MAX_NEW:-64}"
BUDGETS="${BUDGETS:-256 512 1024 2048}"
OURS_SCORER="${OURS_SCORER:-hybrid_late_mmr}"

LONG_TASKS="${LONG_TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,gov_report,multi_news}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot}"
METHODS="${METHODS:-full_kv,streamingllm_sink_recent,h2o_observe,snapkv_observe,ours_page_gather}"

mkdir -p outputs/logs

for BUDGET in $BUDGETS; do
  OUT="outputs/controlled_public_kv_benchmark_v2_expanded_${SAMPLES}shot_b${BUDGET}_${STAMP}"
  LOG="outputs/logs/controlled_public_kv_benchmark_v2_expanded_${SAMPLES}shot_b${BUDGET}_${STAMP}.log"
  python src/run_controlled_public_kv_benchmark_v1.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --benchmarks longbench,ruler \
    --longbench_tasks "$LONG_TASKS" \
    --ruler_tasks "$RULER_TASKS" \
    --max_samples_per_task "$SAMPLES" \
    --max_context_tokens "$MAX_CONTEXT" \
    --max_new_tokens_override "$MAX_NEW" \
    --methods "$METHODS" \
    --budget_tokens "$BUDGET" \
    --sink_tokens 64 \
    --recent_tokens 256 \
    --page_tokens 256 \
    --ruler_lengths 4096 \
    --ours_scorer "$OURS_SCORER" \
    --dtype float16 \
    --attn_implementation eager \
    2>&1 | tee "$LOG"
done
