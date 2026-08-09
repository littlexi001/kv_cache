#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_public_selectors_longbench_official_middle_5gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=full_kv,qksieve_fullprompt_auto_plain_fulltopk,quest_p16_fullprompt_matchedbudget,rabitqcache_rtn1_fullprompt_matchedbudget,sparq_r32_selector_fullprompt_matchedbudget,sparq_r32_formula_fullprompt_matchedbudget
MAX_SAMPLES_PER_TASK="${QKSIEVE_MAX_SAMPLES_PER_TASK:-0}"
if [[ -n "${QKSIEVE_EXPECTED_PAIRS:-}" ]]; then
  EXPECTED_PAIRS="$QKSIEVE_EXPECTED_PAIRS"
elif [[ "$MAX_SAMPLES_PER_TASK" -eq 0 ]]; then
  EXPECTED_PAIRS=3750
else
  EXPECTED_PAIRS=$((16 * MAX_SAMPLES_PER_TASK))
fi

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4}"
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
  --methods "$METHODS" \
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
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

expected = {
    "full_kv",
    "qksieve_fullprompt_auto_plain_fulltopk",
    "quest_p16_fullprompt_matchedbudget",
    "rabitqcache_rtn1_fullprompt_matchedbudget",
    "sparq_r32_selector_fullprompt_matchedbudget",
    "sparq_r32_formula_fullprompt_matchedbudget",
}
with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 6, len(rows)
assert {row["method"] for row in rows} == expected
assert len({(row["task"], row["sample_id"]) for row in rows}) == 1
for row in rows:
    if row["method"] == "full_kv":
        continue
    assert row["executed_path"] == row["method"]
    assert row["configured_attention_tokens"] not in {"", "0"}
print("QKSieve public-selector smoke passed")
PY

pids=()
for ((shard = 0; shard < shard_count; shard++)); do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
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
    --qk_metric_query_shrinkage 0.75 \
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
  echo "one or more public-selector shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" \
  src/analyze_qksieve_public_selectors_longbench_20260728.py \
  --run_root "$RUN_ROOT" \
  --expected_pairs "$EXPECTED_PAIRS" \
  --output "$RUN_ROOT/public_selector_summary.json" \
  >"$LOG_ROOT/summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
