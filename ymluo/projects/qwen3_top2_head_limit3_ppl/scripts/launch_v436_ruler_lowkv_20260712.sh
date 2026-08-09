#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
POLICY="${POLICY:-configs/riskkv_task_policy_v436_ruler_lowkv_b224_20260712.json}"
GPU="${GPU:-auto}"
GPUS="${GPUS:-4,5,0,2,1,3,6,7}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2500}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-25}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_single_2,niah_multikey_1,niah_multivalue,niah_multiquery,cwe,fwe,qa_squad,qa_hotpot,vt}"
RULER_LENGTHS="${RULER_LENGTHS:-4096,8192,16384}"
SAMPLES="${SAMPLES:-50}"
OUT="${OUT:-outputs/riskkv_v436_ruler_lowkv_b224_m${SAMPLES}_20260712}"
LOG="${LOG:-outputs/logs/riskkv_v436_ruler_lowkv_b224_m${SAMPLES}_20260712.log}"
mkdir -p "$(dirname "$LOG")" "$OUT"

if [[ -f "$OUT/task_results.csv" ]]; then
  echo "SKIP existing $OUT/task_results.csv"
  exit 0
fi

choose_gpu() {
  while true; do
    IFS=',' read -ra ids <<< "$GPUS"
    for candidate in "${ids[@]}"; do
      candidate="${candidate//[[:space:]]/}"
      [[ -z "$candidate" ]] && continue
      read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$candidate" | tr -d ',' | awk '{print $1, $2}')
      if [[ "${used:-99999}" -lt "$GPU_MAX_USED_MB" && "${util:-100}" -lt "$GPU_MAX_UTIL" ]]; then
        echo "$candidate"
        return 0
      fi
    done
    local status
    status="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',' | awk '{printf "gpu=%s used=%s util=%s; ", $1, $2, $3}')"
    echo "WAIT_GPU ${status}$(date -Is)" >&2
    sleep 120
  done
}

if [[ "$GPU" == "auto" ]]; then
  GPU="$(choose_gpu)"
fi

echo "LAUNCH v436 RULER low-KV samples=$SAMPLES lengths=$RULER_LENGTHS gpu=$GPU policy=$POLICY"
nohup bash -lc "
  source /home/fdong/miniconda3/etc/profile.d/conda.sh
  conda activate moe
  cd '$ROOT'
  CUDA_VISIBLE_DEVICES='$GPU' '$PY' src/run_controlled_public_kv_benchmark_v1.py \
    --model_name_or_path '$MODEL' \
    --output_dir '$OUT' \
    --benchmarks ruler \
    --ruler_tasks '$RULER_TASKS' \
    --ruler_lengths '$RULER_LENGTHS' \
    --max_samples_per_task '$SAMPLES' \
    --max_context_tokens 20000 \
    --prefill_chunk_tokens 2048 \
    --methods ours_page_gather \
    --budget_tokens 224 \
    --sink_tokens 32 \
    --recent_tokens 32 \
    --page_tokens 64 \
    --ours_scorer hybrid_late_mmr_multiscale_flow \
    --ours_multiscale_group_pages 4 \
    --ours_multiscale_weight 0.22 \
    --ours_flow_neighbor_radius 1 \
    --ours_flow_neighbor_budget_fraction 0.16 \
    --ours_flow_neighbor_min_score 0.12 \
    --ours_flow_score_smooth_weight 0.16 \
    --ours_flow_anchor_boost 0.20 \
    --ours_task_policy_json '$POLICY' \
    --dtype float16 \
    --attn_implementation sdpa \
    --prompt_wrapper llama3 \
    --log_every 20
" > "$LOG" 2>&1 &

echo "Started $OUT"
