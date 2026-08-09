#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SAMPLES="${SAMPLES:-6}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
STAMP="${STAMP:-20260709_task_policy_v19_one}"
LABEL="${LABEL:-safe_budget}"
POLICY="${POLICY:-configs/riskkv_task_policy_v19_safe_budget_20260709.json}"
LOG_ROOT="outputs/logs"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-1000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-15}"
mkdir -p "$LOG_ROOT"
mkdir -p "$LOCK_ROOT"

choose_gpu() {
  while true; do
    IFS=',' read -ra ids <<< "$GPUS"
    for gpu in "${ids[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      [[ -z "$gpu" ]] && continue
      read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ',' | awk '{print $1, $2}')
      if [[ "${used:-99999}" -lt "$GPU_MAX_USED_MB" && "${util:-100}" -lt "$GPU_MAX_UTIL" ]]; then
        local lock_dir="$LOCK_ROOT/gpu${gpu}.lock"
        if mkdir "$lock_dir" 2>/dev/null; then
          echo "$$" > "$lock_dir/pid"
          echo "$gpu"
          return 0
        fi
        local lock_pid=""
        [[ -f "$lock_dir/pid" ]] && lock_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
        if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
          rm -rf "$lock_dir"
        fi
      fi
    done
    local status
    status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',' | awk '{printf "gpu=%s used=%s util=%s; ", $1, $2, $3}')"
    echo "WAIT_GPU_ANY ${status}$(date -Is)" >&2
    sleep 120
  done
}

GPU="$(choose_gpu)"
GPU_LOCK_DIR="$LOCK_ROOT/gpu${GPU}.lock"
trap 'rm -rf "$GPU_LOCK_DIR"' EXIT
OUT="outputs/riskkv_v19_${LABEL}_${STAMP}_m${SAMPLES}_bDyn_pDyn"
LOG="$LOG_ROOT/riskkv_v19_${LABEL}_${STAMP}_m${SAMPLES}_bDyn_pDyn.log"
mkdir -p "$OUT"

echo "START v19_task_policy_one label=$LABEL policy=$POLICY gpu=$GPU samples=$SAMPLES tasks=$TASKS $(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" python src/run_controlled_public_kv_benchmark_v1.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
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
  --ours_scorer hybrid_late_mmr_multiscale_flow \
  --ours_multiscale_group_pages 4 \
  --ours_multiscale_weight 0.22 \
  --ours_flow_neighbor_radius 1 \
  --ours_flow_neighbor_budget_fraction 0.16 \
  --ours_flow_neighbor_min_score 0.12 \
  --ours_flow_score_smooth_weight 0.16 \
  --ours_flow_anchor_boost 0.20 \
  --ours_bridge_budget_fraction 0.16 \
  --ours_bridge_min_score 0.0 \
  --ours_bridge_max_terms 24 \
  --ours_bridge_tasks qasper,musique \
  --ours_task_policy_json "$POLICY" \
  --dtype float16 \
  --attn_implementation sdpa \
  --prompt_wrapper llama3 \
  --longbench_zip_path "$LB_ZIP" \
  --log_every 20 \
  > "$LOG" 2>&1
echo "DONE v19_task_policy_one label=$LABEL $(date -Is)"
cat "$OUT/summary.csv"
