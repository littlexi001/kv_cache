#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl

MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
SAMPLES="${SAMPLES:-3}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt}"
RULER_LENGTHS="${RULER_LENGTHS:-4096,8192}"
MAX_CONTEXT="${MAX_CONTEXT:-20000}"
MAX_NEW="${MAX_NEW:-64}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAMP="${STAMP:-20260709_v93_ruler_scaling}"
METHODS="${METHODS:-full_kv,ours_page_gather}"
POLICY="${POLICY:-configs/riskkv_task_policy_v93_ruler_certificate_scaling_20260709.json}"
LOG_ROOT="outputs/logs"
LOCK_ROOT="${LOCK_ROOT:-/tmp/riskkv_gpu_locks_${USER:-user}}"
mkdir -p "$LOG_ROOT" "$LOCK_ROOT"

choose_gpu() {
  while true; do
    IFS=',' read -ra ids <<< "$GPUS"
    for gpu in "${ids[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      [[ -z "$gpu" ]] && continue
      read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ',' | awk '{print $1, $2}')
      if [[ "${used:-99999}" -lt 1000 && "${util:-100}" -lt 20 ]]; then
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
    sleep 60
  done
}

GPU="$(choose_gpu)"
GPU_LOCK_DIR="$LOCK_ROOT/gpu${GPU}.lock"
trap 'rm -rf "$GPU_LOCK_DIR"' EXIT

OUT="outputs/riskkv_v93_ruler_scaling_${STAMP}_m${SAMPLES}"
LOG="$LOG_ROOT/riskkv_v93_ruler_scaling_${STAMP}_m${SAMPLES}.log"
mkdir -p "$OUT"

echo "START v93_ruler_scaling gpu=$GPU samples=$SAMPLES tasks=$RULER_TASKS lengths=$RULER_LENGTHS methods=$METHODS $(date -Is)"
CUDA_VISIBLE_DEVICES="$GPU" python src/run_controlled_public_kv_benchmark_v1.py \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUT" \
  --benchmarks ruler \
  --ruler_tasks "$RULER_TASKS" \
  --ruler_lengths "$RULER_LENGTHS" \
  --max_samples_per_task "$SAMPLES" \
  --max_context_tokens "$MAX_CONTEXT" \
  --max_new_tokens_override "$MAX_NEW" \
  --methods "$METHODS" \
  --budget_tokens 1024 \
  --sink_tokens 64 \
  --recent_tokens 64 \
  --page_tokens 128 \
  --ours_scorer hybrid_late_mmr_multiscale_idf_flow \
  --ours_idf_mix 0.85 \
  --ours_multiscale_group_pages 4 \
  --ours_multiscale_weight 0.28 \
  --ours_flow_neighbor_radius 1 \
  --ours_flow_neighbor_budget_fraction 0.12 \
  --ours_flow_neighbor_min_score 0.05 \
  --ours_flow_score_smooth_weight 0.12 \
  --ours_flow_anchor_boost 0.25 \
  --ours_task_policy_json "$POLICY" \
  --dtype float16 \
  --attn_implementation sdpa \
  --prompt_wrapper llama3 \
  --log_every 10 \
  > "$LOG" 2>&1
echo "DONE v93_ruler_scaling $(date -Is)"
cat "$OUT/summary.csv"
