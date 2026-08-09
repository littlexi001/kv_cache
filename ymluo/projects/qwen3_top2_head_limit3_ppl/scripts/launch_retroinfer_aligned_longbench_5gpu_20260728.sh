#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/retroinfer/bin/python}"
CHECKOUT="${QKSIEVE_RETROINFER_CHECKOUT:-$ROOT/external/RetrievalAttention}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_retroinfer_aligned_longbench_5gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=retroinfer_stack_full_flash,retroinfer_official_aligned
MAX_SAMPLES_PER_TASK="${QKSIEVE_MAX_SAMPLES_PER_TASK:-0}"
if [[ -n "${QKSIEVE_EXPECTED_PAIRS:-}" ]]; then
  EXPECTED_PAIRS="$QKSIEVE_EXPECTED_PAIRS"
elif [[ "$MAX_SAMPLES_PER_TASK" -eq 0 ]]; then
  EXPECTED_PAIRS=3750
else
  EXPECTED_PAIRS=$((16 * MAX_SAMPLES_PER_TASK))
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

"$PYTHON" -u src/audit_retroinfer_official_checkout_20260728.py \
  --checkout "$CHECKOUT" \
  --output "$RUN_ROOT/official_checkout_audit.json" \
  >"$LOG_ROOT/audit.log" 2>&1

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  echo "QKSIEVE_GPUS must contain exactly five comma-separated GPU ids" >&2
  exit 2
fi
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
  src/run_retroinfer_aligned_longbench_20260728.py \
  --official_checkout "$CHECKOUT" \
  --model_name_or_path "$MODEL" \
  --official_config_model_name meta-llama/Llama-3.1-8B-Instruct \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHODS" \
  --max_samples_per_task 1 \
  --max_prompt_tokens 7500 \
  --max_new_tokens_override 8 \
  --dtype float16 \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 2, len(rows)
assert {row["method"] for row in rows} == {
    "retroinfer_stack_full_flash",
    "retroinfer_official_aligned",
}
assert len({row["prompt_sha256"] for row in rows}) == 1
assert all(row["protocol"] == "qksieve_aligned_longbench_v1" for row in rows)
print("Aligned RetroInfer smoke passed")
PY

pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_retroinfer_aligned_longbench_20260728.py \
    --official_checkout "$CHECKOUT" \
    --model_name_or_path "$MODEL" \
    --official_config_model_name meta-llama/Llama-3.1-8B-Instruct \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
    --num_shards 5 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --dtype float16 \
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
  echo "one or more aligned RetroInfer shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" -u src/analyze_retroinfer_aligned_longbench_20260728.py \
  --run_root "$RUN_ROOT" \
  --expected_pairs "$EXPECTED_PAIRS" \
  --output "$RUN_ROOT/aligned_summary.json" \
  >"$LOG_ROOT/summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
