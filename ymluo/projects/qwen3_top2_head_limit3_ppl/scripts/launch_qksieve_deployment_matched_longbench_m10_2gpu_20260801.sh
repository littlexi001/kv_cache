#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
TEMPLATE="${TEMPLATE:-${ROOT}/results/20260730_qksieve_global_keymse_template_3gpu/templates/global32_3domain_keymse_runtime.pt}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${ROOT}/results/20260801_qksieve_public_selectors_rabitq_longbench_m10_4gpu}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/results/20260801_qksieve_deployment_matched_longbench_m10_2gpu}"
METHOD=qksieve_global_qkbalanced_keymse_wmma_sampled
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "${RUN_ROOT}/logs"
cd "${ROOT}"
test -f "${TEMPLATE}"

IFS=',' read -r -a gpus <<<"${QKSIEVE_GPUS:-4,5}"
if [[ "${#gpus[@]}" -ne 2 ]]; then
  echo "this launcher requires two GPUs" >&2
  exit 2
fi
for gpu in "${gpus[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-5]$ ]]; then
    echo "GPU ${gpu} is outside the allowed 0-5 range" >&2
    exit 2
  fi
done

common_args=(
  --model_name_or_path "${MODEL}"
  --longbench_data_dir "${DATA}"
  --tasks "${TASKS}"
  --methods "${METHOD}"
  --max_prompt_tokens 7500
  --prompt_truncation_mode official_middle
  --official_query_tail_tokens 8
  --max_context_tokens 0
  --prefill_chunk_tokens 2048
  --prompt_wrapper llama3
  --qk_metric_query_shrinkage 0.75
  --packed_qmse_template_in "${TEMPLATE}"
  --sampled_quantile_sample_count 256
  --sampled_quantile_target_tail_count 64
  --dtype float16
  --device cuda
  --device_map auto
)

CUDA_VISIBLE_DEVICES="${gpus[0]}" "${PYTHON}" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  "${common_args[@]}" \
  --output_dir "${RUN_ROOT}/smoke" \
  --tasks narrativeqa \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_new_tokens_override 16 \
  >"${RUN_ROOT}/logs/smoke.log" 2>&1

"${PYTHON}" - "${RUN_ROOT}/smoke/sample_results.csv" "${METHOD}" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
assert len(rows) == 1, len(rows)
row = rows[0]
assert row["method"] == sys.argv[2]
assert row["executed_path"] == row["method"]
assert row["configured_score_mode"].endswith(
    "qfused_gqa4_wmma_kappend_unbiased_packed_direct"
)
assert float(row["configured_index_bits_per_token"]) == 240.0
assert float(row["sampled_quantile_fallback"]) == 0.0
print("QKSieve deployment LongBench smoke passed")
PY

pids=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" "${PYTHON}" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    "${common_args[@]}" \
    --output_dir "${RUN_ROOT}/shard${shard}" \
    --max_samples_per_task 10 \
    --num_shards 2 \
    --shard_index "${shard}" \
    --max_new_tokens_override 0 \
    >"${RUN_ROOT}/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more deployment shards failed; valid rows remain" >&2
  exit 1
fi

"${PYTHON}" src/analyze_qksieve_deployment_longbench_20260801.py \
  --reference_root "${REFERENCE_ROOT}" \
  --deployment_root "${RUN_ROOT}" \
  --expected_pairs 160 \
  --output "${RUN_ROOT}/deployment_matched_summary.json" \
  >"${RUN_ROOT}/logs/summary.log" 2>&1

touch "${RUN_ROOT}/ALL_COMPLETE"
