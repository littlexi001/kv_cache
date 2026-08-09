#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260801_llama31_qksieve_keymse_template_gpu23/templates/llama31_global32_3domain_keymse_runtime.pt}"
FULL_ROOT="${FULL_ROOT:-${ROOT}/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_qasper_proxymass_resolution_gpu0123}"

TARGETS=(16 32 64 64)
TASKS=(qasper qasper qasper qasper,multi_news,passage_count)
SAMPLES=(50 50 50 50)
TAGS=(proxymass_t16 proxymass_t32 proxymass_t64 proxymass_t64_weak3)
METHOD=qksieve_global_qkbalanced_keymse_wmma_proxymass
SCORE_MODE=pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_kappend_proxymass_unbiased_packed_direct

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"

pids=()
for index in 0 1 2 3; do
  target="${TARGETS[$index]}"
  tasks="${TASKS[$index]}"
  sample_count="${SAMPLES[$index]}"
  tag="${TAGS[$index]}"
  method_root="${RUN_ROOT}/${tag}"
  output_dir="${method_root}/shard0"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${index}" "${PYTHON}" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "${MODEL}" \
    --longbench_data_dir "${DATA}" \
    --tasks "${tasks}" \
    --methods "${METHOD}" \
    --max_samples_per_task "${sample_count}" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --packed_qmse_template_in "${TEMPLATE}" \
    --sampled_quantile_sample_count 256 \
    --sampled_quantile_target_tail_count "${target}" \
    --countcap_direct_fraction_override 0.06 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --max_new_tokens_override 0 \
    --output_dir "${output_dir}" \
    >"${RUN_ROOT}/logs/${tag}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

for index in 0 1 2 3; do
  tasks="${TASKS[$index]}"
  sample_count="${SAMPLES[$index]}"
  tag="${TAGS[$index]}"
  method_root="${RUN_ROOT}/${tag}"
  "${PYTHON}" src/analyze_targeted_sparse_longbench_20260801.py \
    --full_root "${FULL_ROOT}" \
    --sparse_root "${method_root}" \
    --method "${METHOD}" \
    --score_mode "${SCORE_MODE}" \
    --tasks "${tasks}" \
    --expected_per_task "${sample_count}" \
    --output "${method_root}/targeted_summary.json" \
    >"${RUN_ROOT}/logs/${tag}_summary.log" 2>&1
done

touch "${RUN_ROOT}/ALL_COMPLETE"
