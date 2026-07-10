#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SAMPLES="${SAMPLES:-6}"
TASKS="${TASKS:-2wikimqa,hotpotqa,musique,qasper,lcc,passage_retrieval_en}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
STAMP="${STAMP:-20260709_task_bridge_v18_m6}"
LOG_ROOT="outputs/logs"
mkdir -p "$LOG_ROOT"

choose_gpu() {
  while true; do
    IFS=',' read -ra ids <<< "$GPUS"
    for gpu in "${ids[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      [[ -z "$gpu" ]] && continue
      read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ',' | awk '{print $1, $2}')
      if [[ "${used:-99999}" -lt 1000 && "${util:-100}" -lt 15 ]]; then
        echo "$gpu"
        return 0
      fi
    done
    local status
    status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',' | awk '{printf "gpu=%s used=%s util=%s; ", $1, $2, $3}')"
    echo "WAIT_GPU_ANY ${status}$(date -Is)" >&2
    sleep 120
  done
}

run_one() {
  local label="$1"
  local bridge_tasks="$2"
  local bridge_fraction="$3"
  local gpu
  gpu="$(choose_gpu)"
  local out="outputs/riskkv_${label}_${STAMP}_b512_p128"
  local log="$LOG_ROOT/riskkv_${label}_${STAMP}_b512_p128.log"
  mkdir -p "$out"
  echo "START label=$label bridge_tasks=$bridge_tasks bridge_fraction=$bridge_fraction gpu=$gpu samples=$SAMPLES tasks=$TASKS $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" python src/run_controlled_public_kv_benchmark_v1.py \
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
    --ours_scorer hybrid_late_mmr_multiscale_task_bridge_flow \
    --ours_multiscale_group_pages 4 \
    --ours_multiscale_weight 0.22 \
    --ours_flow_neighbor_radius 1 \
    --ours_flow_neighbor_budget_fraction 0.16 \
    --ours_flow_neighbor_min_score 0.12 \
    --ours_flow_score_smooth_weight 0.16 \
    --ours_flow_anchor_boost 0.20 \
    --ours_bridge_budget_fraction "$bridge_fraction" \
    --ours_bridge_min_score 0.0 \
    --ours_bridge_max_terms 24 \
    --ours_bridge_tasks "$bridge_tasks" \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --longbench_zip_path "$LB_ZIP" \
    --log_every 12 \
    > "$log" 2>&1
  echo "DONE label=$label bridge_tasks=$bridge_tasks bridge_fraction=$bridge_fraction $(date -Is)"
  cat "$out/summary.csv"
}

run_one "fast_v18_task_bridge_no2wiki_lcc" "hotpotqa,musique,qasper,passage_retrieval_en" 0.16
run_one "fast_v18_task_bridge_qasper_musique" "qasper,musique" 0.16
run_one "fast_v18_task_bridge_no2wiki_lcc_light" "hotpotqa,musique,qasper,passage_retrieval_en" 0.10
