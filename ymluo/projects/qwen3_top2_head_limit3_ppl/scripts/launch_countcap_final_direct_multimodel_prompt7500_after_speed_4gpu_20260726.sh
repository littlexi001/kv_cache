#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
LLAMA_MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
QWEN_MODEL=/home/fdong/models/Qwen3-4B-Instruct
METHOD=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
SAMPLES=100
PARENT_RUN=$ROOT/results/20260726_countcap_final_long_speed_multimodel_4gpu
RUN_ROOT=$ROOT/results/20260726_final_direct_multimodel_m100_prompt7500
LOG_ROOT=$RUN_ROOT/logs
BASELINES=$ROOT/outputs/kvcache_factory_aligned_b1024_20260713_m100_v1/analysis/method_summary.csv

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export COUNTCAP_INPLACE_CACHE_MIN_TOKENS=14000

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued behind $PARENT_RUN"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if ! pgrep -f \
    "launch_countcap_final_long_speed_after_budget_4gpu_20260726.sh|resume_countcap_final_long_speed_after_qwen128_oom_4gpu_20260726.sh" \
    >/dev/null; then
    log "parent stopped without ALL_COMPLETE"
    exit 1
  fi
  sleep 300
done

PIDS=()
LABELS=()

launch_worker() {
  local gpu="$1"
  local model_label="$2"
  local model_path="$3"
  local prompt_wrapper="$4"
  local shard="$5"
  local output_dir="$RUN_ROOT/$model_label/shard$shard"
  local worker_log="$LOG_ROOT/${model_label}_shard${shard}.log"

  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$model_path" \
    --longbench_data_dir "$DATA" \
    --output_dir "$output_dir" \
    --tasks "$TASKS" \
    --methods "full_kv,$METHOD" \
    --max_samples_per_task "$SAMPLES" \
    --num_shards 2 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_prompt_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper "$prompt_wrapper" \
    --dtype float16 --device cuda --device_map auto \
    >"$worker_log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("${model_label}_shard${shard}")
  log "launched ${model_label} shard=${shard} gpu=${gpu} pid=$!"
}

launch_worker 0 llama31_8b "$LLAMA_MODEL" llama3 0
launch_worker 1 llama31_8b "$LLAMA_MODEL" llama3 1
launch_worker 2 qwen3_4b "$QWEN_MODEL" qwen3 0
launch_worker 3 qwen3_4b "$QWEN_MODEL" qwen3 1

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    log "completed ${LABELS[$index]}"
  else
    status=$?
    log "failed ${LABELS[$index]} status=${status}; preserving rows"
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

for model_label in llama31_8b qwen3_4b; do
  "$PYTHON" src/summarize_countcap_benchmark_20260722.py \
    --kind longbench \
    --input_glob "$RUN_ROOT/$model_label/shard*/sample_results.csv" \
    --output_dir "$RUN_ROOT/$model_label/merged" \
    >"$LOG_ROOT/${model_label}_summary.log" 2>&1

  "$PYTHON" - \
    "$RUN_ROOT/$model_label/merged/sample_results.csv" \
    "$TASKS" "$SAMPLES" "$METHOD" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

csv_path, tasks_csv, sample_limit, sparse_method = sys.argv[1:]
tasks = [task for task in tasks_csv.split(",") if task]
expected_pairs = len(tasks) * int(sample_limit)
expected_methods = {"full_kv", sparse_method}

with open(csv_path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

prompt_lengths = [int(float(row["prompt_tokens"])) for row in rows]
assert len(rows) == 2 * expected_pairs
assert set(row["task"] for row in rows) == set(tasks)
assert counts == Counter(
    {method: expected_pairs for method in expected_methods}
)
assert len(pairs) == expected_pairs
assert all(methods == expected_methods for methods in pairs.values())
assert max(prompt_lengths) <= 7500
print(
    f"validated {expected_pairs} strict pairs, {len(tasks)} tasks, "
    f"max_prompt={max(prompt_lengths)}"
)
PY
  touch "$RUN_ROOT/$model_label/ALL_COMPLETE"
done

"$PYTHON" src/summarize_final_direct_multimodel_comparison_20260726.py \
  --llama_csv "$RUN_ROOT/llama31_8b/merged/sample_results.csv" \
  --qwen_csv "$RUN_ROOT/qwen3_4b/merged/sample_results.csv" \
  --baseline_method_summary "$BASELINES" \
  --output_dir "$RUN_ROOT/comparison" \
  --bootstrap_samples 5000 \
  >"$LOG_ROOT/comparison.log" 2>&1

touch "$RUN_ROOT/comparison/ALL_COMPLETE"
touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
