#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260723_countcap_llama31_8b_longbench_full_8gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

wait_group() {
  local failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    log "one or more LongBench shards failed; inspect $LOG_ROOT"
    return 1
  fi
}

if [[ ! -d "$MODEL" ]]; then
  log "missing model: $MODEL"
  exit 2
fi
if [[ ! -d "$DATA" ]]; then
  log "missing LongBench data: $DATA"
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  log "GPU compute processes already exist; refusing to mix timing runs"
  nvidia-smi
  exit 3
fi

log "starting full LongBench on GPUs 0-7: 16 tasks, 3750 paired samples"
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods full_kv,countcap \
    --max_samples_per_task 0 \
    --num_shards 8 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
wait_group

log "all shards complete; merging and validating paired results"
"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  > "$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
with open(path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected_methods = {"full_kv", "countcap"}
counts = Counter(row["method"] for row in rows)
tasks = {row["task"] for row in rows}
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 7500, f"expected 7500 rows, found {len(rows)}"
assert len(tasks) == 16, f"expected 16 tasks, found {len(tasks)}"
assert counts == Counter({"full_kv": 3750, "countcap": 3750}), counts
bad = [key for key, methods in pairs.items() if methods != expected_methods]
assert not bad, f"unpaired samples: {bad[:5]}"
print(f"validated {len(pairs)} paired samples across {len(tasks)} tasks")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/merged/summary.json"
