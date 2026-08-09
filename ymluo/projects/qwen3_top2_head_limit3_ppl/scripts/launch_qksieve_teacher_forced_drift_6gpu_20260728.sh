#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_teacher_forced_drift_32k_4k_6gpu}"
HISTORY_TOKENS="${HISTORY_TOKENS:-32000}"
STEPS="${STEPS:-4096}"
LOG_ROOT="$RUN_ROOT/logs"
TRACE_ROOT="$RUN_ROOT/traces"
ANALYSIS_ROOT="$RUN_ROOT/analysis"
TOPICS=(computer sports medicine space politics religion)

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$LOG_ROOT" "$TRACE_ROOT" "$ANALYSIS_ROOT"
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
for index in 0 1 2 3 4 5; do
  topic=${TOPICS[$index]}
  CUDA_VISIBLE_DEVICES=${gpus[$index]} "$PYTHON" -u \
    src/collect_qksieve_teacher_forced_drift_20260728.py \
    --model_name_or_path "$MODEL" \
    --output_path "$TRACE_ROOT/${topic}.pt" \
    --topic "$topic" \
    --history_tokens "$HISTORY_TOKENS" \
    --steps "$STEPS" \
    --record_steps 0,1,3,7,15,31,63,127,255,511,1023,2047,4095 \
    --layers 0,8,16,24,31 \
    --production_query_tokens 8 \
    --recorded_query_tokens 32 \
    --query_shrinkage 0.75 \
    --prefill_chunk_tokens 2048 \
    --seed 20260728 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/${topic}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more teacher-forced drift workers failed; traces remain" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/analyze_qksieve_query_drift_20260728.py \
  --trace "$TRACE_ROOT/*.pt" \
  --output_dir "$ANALYSIS_ROOT" \
  --query_sample_counts 1,4,8,16,32 \
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
assert coverage["trace_count"] == 6
assert coverage["contains_teacher_forced_continuation"] is True
assert coverage["contains_free_generation"] is False
assert coverage["max_observed_step"] == 4095
assert coverage["covers_1k_decode_query"] is True
assert coverage["covers_2k_decode_query"] is True
assert coverage["covers_4k_decode_query"] is True
assert summary["protocol"]["production_query_samples"] == 8
assert summary["protocol"]["query_sample_counts"] == [1, 4, 8, 16, 32]
assert summary["protocol"]["reserved_physical_index_bits"] == 240
assert summary["counts"]["per_query_rows"] > 0
assert summary["counts"]["per_head_bucket_rows"] > 0
assert summary["counts"]["allocation_rows"] > 0
print("QKSieve 4K teacher-forced Query-drift evidence passed")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
