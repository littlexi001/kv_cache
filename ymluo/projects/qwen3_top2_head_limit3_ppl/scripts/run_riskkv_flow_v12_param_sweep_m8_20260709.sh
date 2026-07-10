#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
GPU="${GPU:-5}"
SAMPLES="${SAMPLES:-8}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
STAMP="${STAMP:-20260709_flow_v12_sweep_m8}"
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

run_flow() {
  local label="$1"
  local radius="$2"
  local fraction="$3"
  local min_score="$4"
  local smooth="$5"
  local boost="$6"
  local out="outputs/riskkv_${label}_${STAMP}"
  local log="$LOG_ROOT/riskkv_${label}_${STAMP}.log"
  mkdir -p "$out"
  echo "START label=$label radius=$radius fraction=$fraction min_score=$min_score smooth=$smooth boost=$boost $(date -Is)"
  CUDA_VISIBLE_DEVICES="$GPU" python src/run_controlled_public_kv_benchmark_v1.py \
    --model_name_or_path "$MODEL" \
    --output_dir "$out" \
    --benchmarks longbench \
    --longbench_tasks "$TASKS" \
    --max_samples_per_task "$SAMPLES" \
    --max_context_tokens 7500 \
    --prefill_chunk_tokens 2048 \
    --methods ours_page_gather \
    --budget_tokens 512 \
    --sink_tokens 64 \
    --recent_tokens 64 \
    --page_tokens 128 \
    --ours_scorer hybrid_late_mmr_flow \
    --ours_flow_neighbor_radius "$radius" \
    --ours_flow_neighbor_budget_fraction "$fraction" \
    --ours_flow_neighbor_min_score "$min_score" \
    --ours_flow_score_smooth_weight "$smooth" \
    --ours_flow_anchor_boost "$boost" \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --longbench_zip_path "$LB_ZIP" \
    --log_every 20 \
    > "$log" 2>&1
  echo "DONE label=$label $(date -Is)"
  cat "$out/summary.csv"
}

wait_for_gpu
run_flow "v12_flow_r1_f015_m018" 1 0.15 0.18 0.18 0.22
run_flow "v12_flow_r1_f030_m010" 1 0.30 0.10 0.18 0.22
run_flow "v12_flow_r2_f022_m018" 2 0.22 0.18 0.22 0.22
run_flow "v12_flow_r2_f035_m000" 2 0.35 0.00 0.25 0.18

