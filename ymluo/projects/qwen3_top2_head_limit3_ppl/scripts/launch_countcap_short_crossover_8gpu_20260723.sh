#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
PARENT_RUN=$ROOT/results/20260723_countcap_fullprompt_longbench_m20_8gpu
RUN_ROOT=$ROOT/results/20260723_countcap_short_crossover_8gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,musique,qmsum,lcc,repobench-p
LENGTHS=(2048 4096 6144 8192 12288 16384 24576 32768)
METHODS=full_kv,countcap_fullprompt,countcap_fullprompt_keypca

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued: waiting for the fullprompt m20 ablation"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
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

log "starting eight-point Llama short-context crossover scan"
for gpu in 0 1 2 3 4 5 6 7; do
  length=${LENGTHS[$gpu]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/length${length}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 2 \
    --num_shards 1 --shard_index 0 \
    --max_context_tokens "$length" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/length${length}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "one or more length probes failed; preserving CSV rows for resume"
  exit 1
fi

for length in "${LENGTHS[@]}"; do
  "$PYTHON" src/summarize_countcap_benchmark_20260722.py \
    --kind longbench \
    --input_glob "$RUN_ROOT/length${length}/sample_results.csv" \
    --output_dir "$RUN_ROOT/length${length}/merged" \
    > "$LOG_ROOT/summary_length${length}.log" 2>&1
done

"$PYTHON" - "$RUN_ROOT" "${LENGTHS[@]}" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
expected = {"full_kv", "countcap_fullprompt", "countcap_fullprompt_keypca"}
for length in map(int, sys.argv[2:]):
    path = root / f"length{length}" / "sample_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["method"] for row in rows)
    pairs = defaultdict(set)
    for row in rows:
        pairs[(row["task"], row["sample_id"])].add(row["method"])
    assert len(rows) == 36, (length, len(rows))
    assert counts == Counter({method: 12 for method in expected}), (length, counts)
    assert len({row["task"] for row in rows}) == 6
    assert len(pairs) == 12 and all(methods == expected for methods in pairs.values())
    print(f"validated length={length}: 12 paired examples")
PY

"$PYTHON" src/analyze_countcap_short_crossover_20260723.py \
  --run_root "$RUN_ROOT" \
  > "$LOG_ROOT/crossover_analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT"
