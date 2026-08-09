#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
MODEL=/home/fdong/models/Qwen2.5-7B-Instruct
METHOD=countcap_fullprompt_keypca_direct_qkvfused_qprojscan_qkvsplitauto_cacheauto_prefillindex
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
SAMPLES=100
PARENT_RUN=$ROOT/results/20260726_final_direct_multimodel_m100_prompt7500
RUN_ROOT=$ROOT/results/20260726_countcap_qwen25_7b_longbench_m100_prompt7500
LOG_ROOT=$RUN_ROOT/logs

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export COUNTCAP_INPLACE_CACHE_MIN_TOKENS=14000

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

model_complete() {
  "$PYTHON" - "$MODEL" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "model.safetensors.index.json"
if not (root / "config.json").is_file() or not index_path.is_file():
    raise SystemExit(1)
with index_path.open(encoding="utf-8") as handle:
    index = json.load(handle)
files = {root / name for name in index["weight_map"].values()}
if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files):
    raise SystemExit(1)
PY
}

log "waiting for parent run and complete Qwen2.5-7B download"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  if [[ -f "$PARENT_RUN/FAILED" ]]; then
    log "parent failed"
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  if ! pgrep -f \
    "launch_countcap_final_direct_multimodel_prompt7500_after_speed_4gpu_20260726.sh" \
    >/dev/null; then
    log "parent stopped without ALL_COMPLETE"
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  sleep 300
done

while ! model_complete; do
  if ! pgrep -f "modelscope download Qwen/Qwen2.5-7B-Instruct" >/dev/null; then
    log "model download stopped before all indexed shards were present"
    touch "$RUN_ROOT/FAILED"
    exit 1
  fi
  sleep 60
done

PIDS=()
LABELS=()

launch_worker() {
  local gpu="$1"
  local shard="$2"
  local output_dir="$RUN_ROOT/shard$shard"
  local worker_log="$LOG_ROOT/shard${shard}.log"

  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$output_dir" \
    --tasks "$TASKS" \
    --methods "full_kv,$METHOD" \
    --max_samples_per_task "$SAMPLES" \
    --num_shards 4 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_prompt_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper qwen3 \
    --dtype float16 --device cuda --device_map auto \
    >"$worker_log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("shard${shard}")
  log "launched shard=${shard} gpu=${gpu} pid=$!"
}

launch_worker 0 0
launch_worker 1 1
launch_worker 2 2
launch_worker 3 3

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

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - \
  "$RUN_ROOT/merged/sample_results.csv" \
  "$TASKS" "$SAMPLES" "$METHOD" <<'PY'
import csv
import json
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

with open(
    "/home/fdong/models/Qwen2.5-7B-Instruct/config.json",
    encoding="utf-8",
) as handle:
    config = json.load(handle)
assert config.get("model_type") == "qwen2"
print(
    f"validated {expected_pairs} strict pairs, {len(tasks)} tasks, "
    f"max_prompt={max(prompt_lengths)}, model_type=qwen2"
)
PY

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
