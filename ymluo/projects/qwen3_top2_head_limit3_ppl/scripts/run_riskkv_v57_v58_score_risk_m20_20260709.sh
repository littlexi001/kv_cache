#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}
PY=${PY:-/home/fdong/miniconda3/envs/moe/bin/python}
MODEL=${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}
ZIP=${ZIP:-outputs/table5_question_aware_riskkv_20260708_table5_question_aware_modelscope_v2/longbench_data.zip}
TASKS=${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}
MAX_SAMPLES=${MAX_SAMPLES:-20}

cd "$ROOT"
mkdir -p logs

common_args=(
  --model_name_or_path "$MODEL"
  --benchmarks longbench
  --longbench_tasks "$TASKS"
  --max_samples_per_task "$MAX_SAMPLES"
  --max_context_tokens 7500
  --prefill_chunk_tokens 2048
  --methods ours_page_gather
  --budget_tokens 512
  --sink_tokens 64
  --recent_tokens 64
  --page_tokens 128
  --ours_scorer hybrid_late_mmr_multiscale_flow
  --ours_multiscale_group_pages 4
  --ours_multiscale_weight 0.22
  --ours_flow_neighbor_radius 1
  --ours_flow_neighbor_budget_fraction 0.16
  --ours_flow_neighbor_min_score 0.12
  --ours_flow_score_smooth_weight 0.16
  --ours_flow_anchor_boost 0.20
  --ours_bridge_budget_fraction 0.16
  --ours_bridge_min_score 0.0
  --ours_bridge_max_terms 24
  --ours_bridge_tasks qasper,musique
  --dtype float16
  --attn_implementation sdpa
  --prompt_wrapper llama3
  --longbench_zip_path "$ZIP"
  --log_every 20
)

CUDA_VISIBLE_DEVICES=${GPU_V57:-0} nohup "$PY" src/run_controlled_public_kv_benchmark_v1.py \
  "${common_args[@]}" \
  --output_dir "outputs/riskkv_v57_predecode_score_risk_full_m20_20260709" \
  --ours_task_policy_json configs/riskkv_task_policy_v57_predecode_score_risk_full_20260709.json \
  > logs/riskkv_v57_predecode_score_risk_full_m20_20260709.log 2>&1 < /dev/null &
echo "V57_SCORE_RISK_FULL_M20_PID=$!"

CUDA_VISIBLE_DEVICES=${GPU_V58:-1} nohup "$PY" src/run_controlled_public_kv_benchmark_v1.py \
  "${common_args[@]}" \
  --output_dir "outputs/riskkv_v58_predecode_score_risk_2048_m20_20260709" \
  --ours_task_policy_json configs/riskkv_task_policy_v58_predecode_score_risk_2048_20260709.json \
  > logs/riskkv_v58_predecode_score_risk_2048_m20_20260709.log 2>&1 < /dev/null &
echo "V58_SCORE_RISK_2048_M20_PID=$!"
