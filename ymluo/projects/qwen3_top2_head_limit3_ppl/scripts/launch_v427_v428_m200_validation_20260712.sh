#!/usr/bin/env bash
set -euo pipefail

source /home/fdong/miniconda3/etc/profile.d/conda.sh
conda activate moe

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
cd "$ROOT"

PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
LB_ZIP="${LB_ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}"
TASKS="${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}"
SAMPLES="${SAMPLES:-200}"
LOG_ROOT="${LOG_ROOT:-outputs/logs}"
mkdir -p "$LOG_ROOT"

launch_ours() {
  local gpu="$1"
  local label="$2"
  local stamp="$3"
  local policy="$4"
  local out="outputs/riskkv_v19_${label}_${stamp}_m${SAMPLES}_bDyn_pDyn"
  local log="$LOG_ROOT/nohup_${label}_${stamp}_m${SAMPLES}.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing $out/task_results.csv"
    return 0
  fi
  echo "LAUNCH ours label=$label samples=$SAMPLES gpu=$gpu policy=$policy"
  nohup env \
    GPUS="$gpu" \
    GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-2500}" \
    GPU_MAX_UTIL="${GPU_MAX_UTIL:-101}" \
    SAMPLES="$SAMPLES" \
    LABEL="$label" \
    STAMP="$stamp" \
    POLICY="$policy" \
    TASKS="$TASKS" \
    LB_ZIP="$LB_ZIP" \
    bash scripts/run_riskkv_task_policy_v19_one_20260709.sh \
    > "$log" 2>&1 &
}

launch_full_raw() {
  local gpu="$1"
  local out="outputs/riskkv_full_kv_longbench_m${SAMPLES}_20260712"
  local log="$LOG_ROOT/riskkv_full_kv_longbench_m${SAMPLES}_20260712.log"
  if [[ -f "$out/task_results.csv" ]]; then
    echo "SKIP existing $out/task_results.csv"
    return 0
  fi
  mkdir -p "$out"
  echo "LAUNCH full_kv samples=$SAMPLES gpu=$gpu"
  nohup bash -lc "
    source /home/fdong/miniconda3/etc/profile.d/conda.sh
    conda activate moe
    cd '$ROOT'
    CUDA_VISIBLE_DEVICES='$gpu' '$PY' src/run_controlled_public_kv_benchmark_v1.py \
      --model_name_or_path '$MODEL' \
      --output_dir '$out' \
      --benchmarks longbench \
      --longbench_tasks '$TASKS' \
      --max_samples_per_task '$SAMPLES' \
      --max_context_tokens 7500 \
      --prefill_chunk_tokens 2048 \
      --methods full_kv \
      --dtype float16 \
      --attn_implementation sdpa \
      --prompt_wrapper llama3 \
      --longbench_zip_path '$LB_ZIP' \
      --log_every 20
  " > "$log" 2>&1 &
}

launch_ours 1 \
  v427_v417_source_v421_winners \
  20260712_v427_m200_validate \
  configs/riskkv_task_policy_v427_v417_source_v421_winners_20260712.json

launch_ours 6 \
  v428_v427_plus_repobench \
  20260712_v428_m200_validate \
  configs/riskkv_task_policy_v428_v427_plus_repobench_20260712.json

launch_full_raw 7

echo "Launched M${SAMPLES} validation jobs. Check outputs/logs/nohup_*_m${SAMPLES}.log and outputs/logs/riskkv_full_kv_longbench_m${SAMPLES}_20260712.log."
