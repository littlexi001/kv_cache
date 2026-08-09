#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_free_generation_drift_6gpu}"
LOG_ROOT="$RUN_ROOT/logs"
ANALYSIS_ROOT="$RUN_ROOT/analysis"
TASKS=narrativeqa,qasper,qmsum,gov_report,multi_news,samsum

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$ANALYSIS_ROOT"
cd "$ROOT"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4,5}"
if [[ "${#gpus[@]}" -ne 6 ]]; then
  echo "QKSIEVE_GPUS must contain exactly six GPU ids" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-5]$ ]]; then
    echo "This protocol is restricted to physical GPUs 0-5; got $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "QKSIEVE_GPUS contains duplicate GPU id $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

pids=()
for shard in 0 1 2 3 4 5; do
  mkdir -p "$RUN_ROOT/shard${shard}/traces"
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods qksieve_fullprompt_auto_plain_fulltopk \
    --max_samples_per_task 4 \
    --num_shards 6 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --qk_trace_output_dir "$RUN_ROOT/shard${shard}/traces" \
    --qk_trace_method qksieve_fullprompt_auto_plain_fulltopk \
    --qk_trace_layers 0,8,16,24,31 \
    --qk_trace_steps 0,1,3,7,15,31,63,127,255,511 \
    --qk_trace_prefill_query_tail_tokens 32 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more free-generation drift workers failed; traces remain" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/analyze_qksieve_query_drift_20260728.py \
  --trace "$RUN_ROOT/shard*/traces/*.pt" \
  --output_dir "$ANALYSIS_ROOT" \
  --query_sample_counts 1,4,8 \
  --production_query_samples 8 \
  --sample_stride 32 \
  --total_rate_budget 15 \
  --query_shrinkage 0.75 \
  --true_top_fraction 0.02 \
  --device cuda \
  >"$LOG_ROOT/analysis.log" 2>&1

"$PYTHON" - "$ANALYSIS_ROOT/summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
coverage = summary["coverage"]
assert summary["schema"] == "qksieve_query_drift_analysis_v1"
assert coverage["contains_free_generation"] is True
assert coverage["contains_teacher_forced_continuation"] is False
assert summary["protocol"]["production_query_samples"] == 8
assert summary["protocol"]["query_sample_counts"] == [1, 4, 8]
assert summary["protocol"]["reserved_physical_index_bits"] == 240
assert summary["counts"]["per_query_rows"] > 0
print(
    "Free-generation coverage is observational:",
    coverage["observed_steps"],
)
PY

touch "$RUN_ROOT/ALL_COMPLETE"
