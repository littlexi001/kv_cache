#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
REFERENCE_ROOT="${REFERENCE_ROOT:-$ROOT/results/20260801_qksieve_public_selectors_longbench_m5_5gpu}"
PROJECTION="${PROJECTION:-$ROOT/data/public_baselines/binarypc/llama3-1-8b-ins-projection-mixlen-mixdata.pt}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260801_qksieve_binarypc_longbench_m5_6gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHOD=binarypc_offline64_fullprompt_matchedbudget
MAX_SAMPLES_PER_TASK="${QKSIEVE_MAX_SAMPLES_PER_TASK:-5}"
EXPECTED_PAIRS=$((16 * MAX_SAMPLES_PER_TASK))

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

if [[ ! -f "$PROJECTION" ]]; then
  echo "BinaryPC projection checkpoint is missing: $PROJECTION" >&2
  exit 2
fi
if [[ ! -d "$REFERENCE_ROOT" ]]; then
  echo "matched reference root is missing: $REFERENCE_ROOT" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4,5}"
if [[ "${#gpus[@]}" -lt 1 || "${#gpus[@]}" -gt 6 ]]; then
  echo "QKSIEVE_GPUS must contain one to six comma-separated GPU ids" >&2
  exit 2
fi
shard_count="${#gpus[@]}"
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-5]$ ]]; then
    echo "QKSIEVE_GPUS is restricted to physical GPUs 0-5; got $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "QKSIEVE_GPUS contains duplicate GPU id $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHOD" \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 7500 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override 8 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --binarypc_projection_path "$PROJECTION" \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 1, len(rows)
row = rows[0]
assert row["method"] == "binarypc_offline64_fullprompt_matchedbudget"
assert row["executed_path"] == row["method"]
assert row["configured_score_mode"] == "binarypc_offline64_fulltopk"
assert float(row["configured_index_bits_per_token"]) == 64.0
assert int(float(row["configured_attention_tokens"])) > 0
print("BinaryPC matched-budget smoke passed")
PY

pids=()
for ((shard = 0; shard < shard_count; shard++)); do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHOD" \
    --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
    --num_shards "$shard_count" \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --binarypc_projection_path "$PROJECTION" \
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
  echo "one or more BinaryPC shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" \
  src/analyze_qksieve_binarypc_longbench_20260801.py \
  --reference_root "$REFERENCE_ROOT" \
  --binarypc_root "$RUN_ROOT" \
  --expected_pairs "$EXPECTED_PAIRS" \
  --projection_path "$PROJECTION" \
  --output "$RUN_ROOT/binarypc_matched_summary.json" \
  >"$LOG_ROOT/summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
