#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
POLICY="${POLICY:-configs/riskkv_task_policy_v427_v417_source_v421_winners_20260712.json}"
RULER_TASKS="${RULER_TASKS:-niah_single_1,niah_single_2,niah_multikey_1,niah_multivalue,niah_multiquery,cwe,fwe,qa_squad,qa_hotpot,vt}"
RULER_LENGTHS="${RULER_LENGTHS:-4096,8192,16384}"
SAMPLES="${SAMPLES:-50}"
LOG_ROOT="${LOG_ROOT:-outputs/logs}"
mkdir -p "$LOG_ROOT"

launch_ours() {
  local gpu="$1"
  local out="outputs/riskkv_v427_ruler_m${SAMPLES}_b384_20260712"
  local log="$LOG_ROOT/riskkv_v427_ruler_m${SAMPLES}_b384_20260712.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing $out/task_results.csv"
    return 0
  fi
  mkdir -p "$out"
  echo "LAUNCH v427 ruler samples=$SAMPLES lengths=$RULER_LENGTHS gpu=$gpu"
  nohup bash -lc "
    source /home/fdong/miniconda3/etc/profile.d/conda.sh
    conda activate moe
    cd '$ROOT'
    CUDA_VISIBLE_DEVICES='$gpu' '$PY' src/run_controlled_public_kv_benchmark_v1.py \
      --model_name_or_path '$MODEL' \
      --output_dir '$out' \
      --benchmarks ruler \
      --ruler_tasks '$RULER_TASKS' \
      --ruler_lengths '$RULER_LENGTHS' \
      --max_samples_per_task '$SAMPLES' \
      --max_context_tokens 20000 \
      --prefill_chunk_tokens 2048 \
      --methods ours_page_gather \
      --budget_tokens 384 \
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
      --ours_task_policy_json '$POLICY' \
      --dtype float16 \
      --attn_implementation sdpa \
      --prompt_wrapper llama3 \
      --log_every 20
  " > "$log" 2>&1 &
}

launch_full_kv() {
  local gpu="$1"
  local out="outputs/riskkv_full_kv_ruler_m${SAMPLES}_20260712"
  local log="$LOG_ROOT/riskkv_full_kv_ruler_m${SAMPLES}_20260712.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing $out/task_results.csv"
    return 0
  fi
  mkdir -p "$out"
  echo "LAUNCH full_kv ruler samples=$SAMPLES lengths=$RULER_LENGTHS gpu=$gpu"
  nohup bash -lc "
    source /home/fdong/miniconda3/etc/profile.d/conda.sh
    conda activate moe
    cd '$ROOT'
    CUDA_VISIBLE_DEVICES='$gpu' '$PY' src/run_controlled_public_kv_benchmark_v1.py \
      --model_name_or_path '$MODEL' \
      --output_dir '$out' \
      --benchmarks ruler \
      --ruler_tasks '$RULER_TASKS' \
      --ruler_lengths '$RULER_LENGTHS' \
      --max_samples_per_task '$SAMPLES' \
      --max_context_tokens 20000 \
      --prefill_chunk_tokens 2048 \
      --methods full_kv \
      --dtype float16 \
      --attn_implementation sdpa \
      --prompt_wrapper llama3 \
      --log_every 20
  " > "$log" 2>&1 &
}

launch_ours 0
launch_full_kv 2

echo "Launched RULER M${SAMPLES} validation jobs."
