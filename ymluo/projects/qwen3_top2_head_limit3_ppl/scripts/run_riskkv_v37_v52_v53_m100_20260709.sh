#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}
ZIP=${ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}
TASKS=${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}
MAX_SAMPLES=${MAX_SAMPLES:-100}

cd "$ROOT"
mkdir -p logs

common_args=(
  --model_name_or_path "$MODEL"
  --benchmarks longbench
  --longbench_tasks "$TASKS"
  --max_samples_per_task "$MAX_SAMPLES"
  --max_context_tokens 7500
  --prefill_chunk_tokens 2048
  --budget_tokens 512
  --sink_tokens 64
  --recent_tokens 64
  --page_tokens 128
  --dtype float16
  --attn_implementation sdpa
  --prompt_wrapper llama3
  --longbench_zip_path "$ZIP"
  --log_every 20
)

run_ours() {
  local gpu=$1
  local name=$2
  local policy=$3
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" src/run_controlled_public_kv_benchmark_v1.py \
    "${common_args[@]}" \
    --output_dir "outputs/${name}" \
    --methods ours_page_gather \
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
    --ours_task_policy_json "$policy" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "${name}_PID=$!"
}

CUDA_VISIBLE_DEVICES=${GPU_FULL:-0} nohup "$PY" src/run_controlled_public_kv_benchmark_v1.py \
  "${common_args[@]}" \
  --output_dir "outputs/riskkv_fullkv_m100_same_samples_20260709" \
  --methods full_kv \
  > logs/riskkv_fullkv_m100_same_samples_20260709.log 2>&1 < /dev/null &
echo "FULL_M100_PID=$!"

run_ours "${GPU_V37:-1}" "riskkv_v37_high_quality_m100_20260709" \
  configs/riskkv_task_policy_v37_high_quality_20260709.json

run_ours "${GPU_V52:-2}" "riskkv_v52_consistency_quality_m100_20260709" \
  configs/riskkv_task_policy_v52_consistency_quality_20260709.json

run_ours "${GPU_V53:-3}" "riskkv_v53_consistency_quality_qasper_full_m100_20260709" \
  configs/riskkv_task_policy_v53_consistency_quality_qasper_full_20260709.json
