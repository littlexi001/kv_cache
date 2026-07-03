#!/usr/bin/env bash
set -euo pipefail

cd /home/fdong/ymluo/external/KVCache-Factory
source /home/fdong/miniconda3/bin/activate moe

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/fdong/ymluo/pydeps/kvcache_factory_tf444:$PWD:${PYTHONPATH:-}"

STAMP="${STAMP:-20260703_official}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
SAMPLES="${SAMPLES:-1}"
BUDGETS="${BUDGETS:-256 512 1024 2048}"
METHODS="${METHODS:-FullKV StreamingLLM H2O SnapKV PyramidKV AdaKV}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,gov_report,multi_news}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-float16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OUT_ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_official_longbench_${SAMPLES}shot_${STAMP}"
LOG_ROOT="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/logs"
STATUS="$OUT_ROOT/run_status.csv"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
echo "budget,method,status,log" > "$STATUS"

for BUDGET in $BUDGETS; do
  for METHOD in $METHODS; do
    LOG="$LOG_ROOT/kvcache_factory_longbench_${METHOD}_b${BUDGET}_${SAMPLES}shot_${STAMP}.log"
    set +e
    python run_longbench.py \
      --method "$METHOD" \
      --model_path "$MODEL" \
      --max_capacity_prompts "$BUDGET" \
      --attn_implementation "$ATTN" \
      --save_dir "$OUT_ROOT" \
      --use_cache True \
      --datasets "$TASKS" \
      --max_num_examples "$SAMPLES" \
      --sample_method topk \
      --dtype "$DTYPE" \
      2>&1 | tee "$LOG"
    STATUS_CODE=${PIPESTATUS[0]}
    set -e
    if [[ "$STATUS_CODE" -eq 0 ]]; then
      echo "$BUDGET,$METHOD,OK,$LOG" >> "$STATUS"
    else
      echo "$BUDGET,$METHOD,FAILED,$LOG" >> "$STATUS"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$STATUS_CODE"
      fi
    fi
  done
done
