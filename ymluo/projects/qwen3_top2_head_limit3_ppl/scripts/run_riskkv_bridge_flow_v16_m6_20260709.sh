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
STAMP="${STAMP:-20260709_bridge_flow_v16_m6}"
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
  local scorer="$2"
  local page="$3"
  local budget="$4"
  local bridge_fraction="$5"
  local bridge_terms="$6"
  local gpu
  gpu="$(choose_gpu)"
  local out="outputs/riskkv_${label}_${STAMP}_b${budget}_p${page}"
  local log="$LOG_ROOT/riskkv_${label}_${STAMP}_b${budget}_p${page}.log"
  mkdir -p "$out"
  echo "START label=$label scorer=$scorer page=$page budget=$budget bridge_fraction=$bridge_fraction bridge_terms=$bridge_terms gpu=$gpu samples=$SAMPLES tasks=$TASKS $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" python src/run_controlled_public_kv_benchmark_v1.py \
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
    --ours_multiscale_group_pages 4 \
    --ours_multiscale_weight 0.22 \
    --ours_flow_neighbor_radius 1 \
    --ours_flow_neighbor_budget_fraction 0.16 \
    --ours_flow_neighbor_min_score 0.12 \
    --ours_flow_score_smooth_weight 0.16 \
    --ours_flow_anchor_boost 0.20 \
    --ours_bridge_budget_fraction "$bridge_fraction" \
    --ours_bridge_min_score 0.0 \
    --ours_bridge_max_terms "$bridge_terms" \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --longbench_zip_path "$LB_ZIP" \
    --log_every 12 \
    > "$log" 2>&1
  echo "DONE label=$label scorer=$scorer page=$page budget=$budget bridge_fraction=$bridge_fraction $(date -Is)"
  cat "$out/summary.csv"
}

run_one "fast_v16_bridge_p128" "hybrid_late_mmr_bridge_flow" 128 512 0.16 24
run_one "fast_v16_ms_bridge_p128" "hybrid_late_mmr_multiscale_bridge_flow" 128 512 0.16 24
run_one "fast_v16_ms_bridge_p64" "hybrid_late_mmr_multiscale_bridge_flow" 64 512 0.18 24
run_one "fast_v16_ms_bridge_p128_wide" "hybrid_late_mmr_multiscale_bridge_flow" 128 768 0.20 32
