#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
FULL_ROOT="${FULL_ROOT:-${ROOT}/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_robust_compare_longbench_m5_6gpu}"
RANK16_METHOD="qksieve_requestlocal_qkbalanced_keymse_wmma_valuesketch16"
MEANTAIL_METHOD="qksieve_requestlocal_qkbalanced_keymse_wmma_meantail"
METHODS="${METHODS:-${RANK16_METHOD},${MEANTAIL_METHOD}}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-5}"
EXPECTED_PER_TASK="${EXPECTED_PER_TASK:-${MAX_SAMPLES_PER_TASK}}"
RANK16_MODE="pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_valuesketch16i4shared_unbiased_packed_direct"
MEANTAIL_MODE="pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_meantail_unbiased_packed_direct"
ALL_TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p"
TASK_GROUPS=(
  "narrativeqa,qasper,multifieldqa_en"
  "hotpotqa,2wikimqa,musique"
  "qmsum,trec,triviaqa"
  "samsum,passage_retrieval_en,passage_count"
  "gov_report,multi_news"
  "lcc,repobench-p"
)

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export QKSIEVE_VALUE_SKETCH_TAIL_ALPHA="0.5"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

pids=()
for gpu in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA}" \
    --tasks "${TASK_GROUPS[$gpu]}" \
    --methods "${METHODS}" \
    --max_samples_per_task "${MAX_SAMPLES_PER_TASK}" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --sampled_quantile_sample_count 256 \
    --sampled_quantile_target_tail_count 32 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_new_tokens_override 0 \
    --output_dir "${RUN_ROOT}/shard${gpu}" \
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

if [[ ",${METHODS}," == *",${RANK16_METHOD},"* ]]; then
  "${PYTHON}" src/analyze_targeted_sparse_longbench_20260801.py \
    --full_root "${FULL_ROOT}" --sparse_root "${RUN_ROOT}" \
    --method "${RANK16_METHOD}" --score_mode "${RANK16_MODE}" \
    --tasks "${ALL_TASKS}" --expected_per_task "${EXPECTED_PER_TASK}" \
    --output "${RUN_ROOT}/summary_rank16.json" \
    >"${RUN_ROOT}/logs/summary_rank16.log" 2>&1
fi

if [[ ",${METHODS}," == *",${MEANTAIL_METHOD},"* ]]; then
  "${PYTHON}" src/analyze_targeted_sparse_longbench_20260801.py \
    --full_root "${FULL_ROOT}" --sparse_root "${RUN_ROOT}" \
    --method "${MEANTAIL_METHOD}" --score_mode "${MEANTAIL_MODE}" \
    --tasks "${ALL_TASKS}" --expected_per_task "${EXPECTED_PER_TASK}" \
    --output "${RUN_ROOT}/summary_meantail.json" \
    >"${RUN_ROOT}/logs/summary_meantail.log" 2>&1
fi

touch "${RUN_ROOT}/ALL_COMPLETE"
