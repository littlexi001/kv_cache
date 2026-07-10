#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
GPU="${GPU:-6}"
SAMPLES="${SAMPLES:-8}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
STAMP="${STAMP:-20260709_multiscale_flow_v13_m8}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

wait_for_gpu() {
  while true; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$GPU" | tr -d ',' | awk '{print $1, $2}')
    if [[ "${used:-99999}" -lt 1000 && "${util:-100}" -lt 15 ]]; then
      echo "GPU_IDLE gpu=$GPU used=$used util=$util $(date -Is)"
      return 0
    fi
    echo "WAIT_GPU gpu=$GPU used=${used:-?} util=${util:-?} $(date -Is)"
    sleep 120
  done
}

run_one() {
  local label="$1"
  local page="$2"
  local budget="$3"
  local group_pages="$4"
  local out="outputs/riskkv_${label}_${STAMP}_b${budget}_p${page}"
  local log="$LOG_ROOT/riskkv_${label}_${STAMP}_b${budget}_p${page}.log"
  mkdir -p "$out"
  echo "START label=$label page=$page budget=$budget group_pages=$group_pages gpu=$GPU samples=$SAMPLES $(date -Is)"
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
    --ours_scorer hybrid_late_mmr_multiscale_flow \
    --ours_multiscale_group_pages "$group_pages" \
    --ours_multiscale_weight 0.22 \
    --ours_flow_neighbor_radius 1 \
    --ours_flow_neighbor_budget_fraction 0.18 \
    --ours_flow_neighbor_min_score 0.12 \
    --ours_flow_score_smooth_weight 0.16 \
    --ours_flow_anchor_boost 0.20 \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --longbench_zip_path "$LB_ZIP" \
    --log_every 20 \
    > "$log" 2>&1
  echo "DONE label=$label page=$page budget=$budget group_pages=$group_pages $(date -Is)"
  cat "$out/summary.csv"
}

wait_for_gpu
run_one "v13_multiscale_flow_p128_g4" 128 512 4
run_one "v13_multiscale_flow_p64_g4" 64 512 4
run_one "v13_multiscale_flow_p64_g8" 64 512 8

