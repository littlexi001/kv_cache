#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
WAIT_ROOT=$ROOT/results/20260723_countcap_auto_longbench_m20_8gpu
RUN_ROOT=$ROOT/results/20260723_countcap_quality_diagnostic_m8_8gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=full_kv,exact_top2_fullprompt,exact_massadaptive_fullprompt,countcap_fullprompt,countcap_massadaptive_fullprompt

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued: waiting for the short-context auto validation"
while [[ ! -f "$WAIT_ROOT/ALL_COMPLETE" ]]; do
  sleep 60
done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  sleep 30
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

log "starting exact-versus-approximate quality diagnostic"
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --mass_threshold 0.95 \
    --sample_fraction 0.01 \
    --collect_attention_stats \
    --max_samples_per_task 8 \
    --num_shards 8 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "one or more diagnostic shards failed; preserving rows for resume"
  exit 1
fi

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  > "$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
expected = {
    "full_kv",
    "exact_top2_fullprompt",
    "exact_massadaptive_fullprompt",
    "countcap_fullprompt",
    "countcap_massadaptive_fullprompt",
}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])
assert len(rows) == 640, len(rows)
assert counts == Counter({method: 128 for method in expected}), counts
assert len({row["task"] for row in rows}) == 16
assert len(pairs) == 128 and all(methods == expected for methods in pairs.values())
PY

"$PYTHON" src/analyze_countcap_quality_diagnostic_20260723.py \
  --run_root "$RUN_ROOT" \
  > "$LOG_ROOT/quality_diagnostic.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/merged/summary.json"
