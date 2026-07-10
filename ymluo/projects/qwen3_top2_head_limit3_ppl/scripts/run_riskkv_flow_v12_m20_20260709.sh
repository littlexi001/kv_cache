#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
GPU="${GPU:-5}"
SAMPLES="${SAMPLES:-20}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
STAMP="${STAMP:-20260709_flow_v12_m20}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

run_one() {
  local label="$1"
  local scorer="$2"
  local budget="$3"
  local page="$4"
  local out="outputs/riskkv_${label}_${STAMP}_b${budget}_p${page}"
  local log="$LOG_ROOT/riskkv_${label}_${STAMP}_b${budget}_p${page}.log"
  mkdir -p "$out"
  echo "START label=$label scorer=$scorer budget=$budget page=$page gpu=$GPU samples=$SAMPLES $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" python src/run_controlled_public_kv_benchmark_v1.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$out" \
    --benchmarks longbench \
    --longbench_tasks "$TASKS" \
    --max_samples_per_task "$SAMPLES" \
    --max_context_tokens 7500 \
    --prefill_chunk_tokens 2048 \
    --methods ours_page_gather \
    --budget_tokens "$budget" \
    --sink_tokens 64 \
    --recent_tokens 64 \
    --page_tokens "$page" \
    --ours_scorer "$scorer" \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --longbench_zip_path "$LB_ZIP" \
    --log_every 20 \
    > "$log" 2>&1
  echo "DONE label=$label scorer=$scorer budget=$budget page=$page $(date -Is)"
  cat "$out/summary.csv"
}

run_one "v11_hybrid_late_mmr" "hybrid_late_mmr" 512 128
run_one "v12_flow" "hybrid_late_mmr_flow" 512 128
run_one "v12_flow_conservative" "hybrid_late_mmr_flow" 1024 128

