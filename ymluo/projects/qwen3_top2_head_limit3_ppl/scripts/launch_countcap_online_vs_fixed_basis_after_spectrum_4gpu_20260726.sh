#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
LLAMA_MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
QWEN_MODEL=/home/fdong/models/Qwen3-4B-Instruct
PARENT_RUN=$ROOT/results/20260726_qk_matrix_spectrum_multimodel_32k
RUN_ROOT=$ROOT/results/20260726_countcap_online_vs_fixed_basis_m20_4gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
SAMPLES=20

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
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    log "parent failed"
    exit 1
  fi
  if ! pgrep -f \
    "launch_qk_matrix_spectrum_multimodel_after_v4_20260726.sh" \
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
    src/run_countcap_online_vs_fixed_basis_longbench_20260726.py \
    --model_name_or_path "$model_path" \
    --longbench_data_dir "$DATA" \
    --output_dir "$output_dir" \
    --tasks "$TASKS" \
    --max_samples_per_task "$SAMPLES" \
    --num_shards 2 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_prompt_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper "$prompt_wrapper" \
    --calibration_specs \
      "gov_report:150,narrativeqa:150,qasper:150,repobench-p:150" \
    --calibration_max_new_tokens 2 \
    --projection_dim 48 \
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
  touch "$RUN_ROOT/FAILED"
  exit 1
fi

"$PYTHON" src/summarize_countcap_online_vs_fixed_basis_20260726.py \
  --llama_glob "$RUN_ROOT/llama31_8b/shard*/sample_results.csv" \
  --qwen_glob "$RUN_ROOT/qwen3_4b/shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/analysis" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT" "$TASKS" "$SAMPLES" <<'PY'
import csv
import glob
import json
import sys
from collections import Counter, defaultdict

root, tasks_csv, sample_limit = sys.argv[1:]
tasks = {task for task in tasks_csv.split(",") if task}
expected_pairs = len(tasks) * int(sample_limit)
expected_methods = {"online", "fixed"}

for model in ("llama31_8b", "qwen3_4b"):
    rows = []
    for path in glob.glob(root + f"/{model}/shard[0-9]*/sample_results.csv"):
        with open(path, encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    counts = Counter(row["method"] for row in rows)
    pairs = defaultdict(set)
    for row in rows:
        pairs[(row["task"], row["sample_id"])].add(row["method"])
    assert len(rows) == 2 * expected_pairs, (model, len(rows))
    assert set(row["task"] for row in rows) == tasks
    assert counts == Counter(
        {method: expected_pairs for method in expected_methods}
    )
    assert len(pairs) == expected_pairs
    assert all(methods == expected_methods for methods in pairs.values())

with open(root + "/analysis/summary.json", encoding="utf-8") as handle:
    summary = json.load(handle)
assert len(summary["overall"]) == 2
assert all(row["tasks"] == 16 for row in summary["overall"])
assert all(row["paired_samples"] == expected_pairs for row in summary["overall"])
print(f"validated two models x {expected_pairs} strict online/fixed pairs")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
