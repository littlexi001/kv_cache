#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
FULL_ROOT="${FULL_ROOT:-${ROOT}/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_requestlocal_proxymass_longbench_m10_gpu0123}"
METHOD=qksieve_requestlocal_qkbalanced_keymse_wmma_proxymass
SCORE_MODE=pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_proxymass_unbiased_packed_direct
ALL_TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
TASK_GROUPS=(
  narrativeqa,qasper,multifieldqa_en,hotpotqa
  2wikimqa,musique,qmsum,trec
  triviaqa,samsum,passage_retrieval_en,passage_count
  gov_report,multi_news,lcc,repobench-p
)

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

pids=()
for gpu in 0 1 2 3; do
  tasks="${TASK_GROUPS[$gpu]}"
  output_dir="${RUN_ROOT}/shard${gpu}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA}" \
    --tasks "${tasks}" \
    --methods "${METHOD}" \
    --max_samples_per_task 10 \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --sampled_quantile_sample_count 256 \
    --sampled_quantile_target_tail_count 32 \
    --countcap_direct_fraction_override 0.06 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_new_tokens_override 0 \
    --output_dir "${output_dir}" \
    >"${RUN_ROOT}/logs/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" src/analyze_targeted_sparse_longbench_20260801.py \
  --full_root "${FULL_ROOT}" \
  --sparse_root "${RUN_ROOT}" \
  --method "${METHOD}" \
  --score_mode "${SCORE_MODE}" \
  --tasks "${ALL_TASKS}" \
  --expected_per_task 10 \
  --output "${RUN_ROOT}/targeted_summary.json" \
  >"${RUN_ROOT}/logs/summary.log" 2>&1
touch "${RUN_ROOT}/ALL_COMPLETE"
