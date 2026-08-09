#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/nanogpt/bin/python}"
MODEL="${MODEL:-/home/fdong/qksieve_iclr2027/models/Qwen3-4B-Instruct-2507}"
OUTPUT="${OUTPUT:-$ROOT/results/tail_shrinkage_realqkv_longbench_20260807}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/LongBench}"

export PATH="/home/fdong/miniconda3/envs/nanogpt/bin:/usr/local/cuda/bin:/usr/bin:/bin"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT/traces" "$OUTPUT/analysis" "$OUTPUT/logs"
cd "$ROOT"

run_case() {
  local visible_gpus="$1"
  local label="$2"
  local source_task="$3"
  local history_tokens="$4"
  local seed="$5"
  local trace="$OUTPUT/traces/${label}.pt"
  local analysis="$OUTPUT/analysis/${label}"

  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PYTHON" -u \
    src/collect_real_qk_trace_20260715.py \
    --model_name_or_path "$MODEL" \
    --output_path "$trace" \
    --source_jsonl "$DATA_ROOT/${source_task}.jsonl" \
    --source_field context \
    --repeat_source_documents \
    --history_tokens "$history_tokens" \
    --steps 1 \
    --layers 8,17,26 \
    --prefill_query_tail_tokens 8 \
    --prefill_chunk_tokens 1024 \
    --seed "$seed" \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$OUTPUT/logs/${label}_capture.log" 2>&1

  CUDA_VISIBLE_DEVICES="$visible_gpus" "$PYTHON" -u \
    src/analyze_qksieve_tail_partition_calibration_20260803.py \
    --traces "$trace" \
    --output_dir "$analysis" \
    --model_name_or_path "$MODEL" \
    --device cuda \
    --top_k 1280 \
    --sample_counts 256 \
    --block_sizes 256 \
    --conditional_dims 8 \
    --conditional_fit_stride 32 \
    --tail_sampling systematic \
    --key_sample_stride 32 \
    --value_sample_stride 32 \
    --query_shrinkage 0.75 \
    --key_rate_budget 15 \
    --value_rank 16 \
    --value_bits 4 \
    --value_scale_block 256 \
    --value_metric wo_group \
    --max_records_per_trace 3 \
    >"$OUTPUT/logs/${label}_analysis.log" 2>&1

  touch "$OUTPUT/${label}_COMPLETE"
}

# The first three cases isolate length on the same NarrativeQA corpus.  LCC and
# QMSum test whether the shrinkage statistic is tied to one task's text style.
run_case 0 narrative32k narrativeqa 32768 20260851 & pid0=$!
run_case 1 narrative64k narrativeqa 65536 20260851 & pid1=$!
run_case 2 lcc64k lcc 65536 20260852 & pid2=$!
run_case 3 qmsum64k qmsum 65536 20260853 & pid3=$!
run_case 4,5 narrative128k narrativeqa 131072 20260851 & pid4=$!

status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
touch "$OUTPUT/ALL_COMPLETE"
