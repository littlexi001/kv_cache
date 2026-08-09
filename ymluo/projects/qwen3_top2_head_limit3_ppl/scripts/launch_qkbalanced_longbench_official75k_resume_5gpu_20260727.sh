#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260727_qkbalanced_longbench_official75k_full_8gpu}"
TASKS="narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p"
METHODS="full_kv,countcap_fullprompt_qkbalanced_packed_direct"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/logs/resume_5gpu.log"
}

run_shard() {
  local gpu="$1"
  local shard="$2"
  log "GPU $gpu resumes shard $shard"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 0 \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$RUN_ROOT/logs/shard${shard}.resume_5gpu.log" 2>&1
}

run_worker() {
  local gpu="$1"
  shift
  local shard
  for shard in "$@"; do
    run_shard "$gpu" "$shard"
  done
}

log "resume on idle physical GPUs 0,1,2,3,7"
pids=()
run_worker 0 7 & pids+=("$!")
run_worker 1 3 & pids+=("$!")
run_worker 2 0 1 & pids+=("$!")
run_worker 3 5 4 & pids+=("$!")
run_worker 7 6 2 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "one or more resumed shards failed; valid CSV rows remain preserved"
  exit 1
fi

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$RUN_ROOT/logs/summary.log" 2>&1

"$PYTHON" src/analyze_qkbalanced_longbench_paired_20260727.py \
  --input_csv "$RUN_ROOT/merged/sample_results.csv" \
  --output_dir "$RUN_ROOT/paired_analysis" \
  >"$RUN_ROOT/logs/paired_analysis.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected = {"full_kv", "countcap_fullprompt_qkbalanced_packed_direct"}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 7500, (len(rows), counts)
assert counts == Counter({method: 3750 for method in expected}), counts
assert len(pairs) == 3750
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(row["prompt_tokens"]) for row in rows) <= 7500
print("validated 3750 strict Full/QK-balanced official-protocol pairs")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
