#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RESEARCH_RUN=$ROOT/results/20260723_countcap_direct_8k16k_4gpu
RUN_ROOT=$ROOT/results/20260723_countcap_adakv_table5_official75k_full_8gpu
LOG_ROOT=$RUN_ROOT/logs
GPUS=(0 1 2 3)
LOGICAL_SHARDS=(4 5 6 7)
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=full_kv,countcap_fullprompt_keypca

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/resume_shards4to7.log"
}

gpus_zero_to_three_idle() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
      return 1
    fi
  done
  return 0
}

log "queued: waiting for direct-attention research on GPUs 0-3"
while [[ ! -f "$RESEARCH_RUN/ALL_COMPLETE" ]]; do
  sleep 30
done
while ! gpus_zero_to_three_idle; do
  sleep 30
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

log "resuming logical shards 4-7 on physical GPUs 0-3"
for slot in 0 1 2 3; do
  gpu=${GPUS[$slot]}
  shard=${LOGICAL_SHARDS[$slot]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 0 \
    --num_shards 8 --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/shard${shard}_resume_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "one or more resumed shards failed; preserving all valid CSV rows"
  exit 1
fi

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  > "$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected = {"full_kv", "countcap_fullprompt_keypca"}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 7500, f"expected 7500 rows, found {len(rows)}"
assert counts == Counter({method: 3750 for method in expected}), counts
assert len(pairs) == 3750
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(row["prompt_tokens"]) for row in rows) <= 7500
print("validated 3750 strict official-protocol Full/CountCap pairs")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/merged/summary.json"
