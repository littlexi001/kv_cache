#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
MAIN_RUN=$ROOT/results/20260723_countcap_llama31_8b_longbench_full_8gpu
RUN_ROOT=$ROOT/results/20260723_countcap_fullprompt_longbench_m20_8gpu
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

log "queued: waiting for the full 3750-sample LongBench run"
while [[ ! -f "$MAIN_RUN/ALL_COMPLETE" ]]; do
  sleep 60
done

log "main run complete; waiting for all GPUs to become idle"
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

log "starting paired m20 short-path ablation on GPUs 0-7"
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods full_kv,countcap_fullprompt,countcap_fullprompt_keypca \
    --max_samples_per_task 20 \
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
  log "one or more shards failed; preserving completed rows for resume"
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

expected = {"full_kv", "countcap_fullprompt", "countcap_fullprompt_keypca"}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 960, f"expected 960 rows, found {len(rows)}"
assert len({row["task"] for row in rows}) == 16
assert counts == Counter({
    "full_kv": 320,
    "countcap_fullprompt": 320,
    "countcap_fullprompt_keypca": 320,
}), counts
assert all(methods == expected for methods in pairs.values())
print(f"validated {len(pairs)} paired samples")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/merged/summary.json"
