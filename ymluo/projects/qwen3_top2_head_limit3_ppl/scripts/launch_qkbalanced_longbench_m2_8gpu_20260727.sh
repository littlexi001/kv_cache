#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260727_qkbalanced_longbench_m2_8gpu}"
METHODS="full_kv,countcap_fullprompt_qkbalanced_packed_direct"

export PATH="/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT/logs"
cd "$ROOT"

task_groups=(
  "narrativeqa,qasper"
  "multifieldqa_en,hotpotqa"
  "2wikimqa,musique"
  "qmsum,trec"
  "triviaqa,samsum"
  "passage_retrieval_en,passage_count"
  "gov_report,multi_news"
  "lcc,repobench-p"
)

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  output="$RUN_ROOT/shard${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$output" \
    --tasks "${task_groups[$gpu]}" \
    --methods "$METHODS" \
    --max_samples_per_task 2 \
    --num_shards 1 \
    --shard_index 0 \
    --max_prompt_tokens 7500 \
    --max_context_tokens 0 \
    --max_new_tokens_override 64 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --collect_attention_stats \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$RUN_ROOT/logs/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$RUN_ROOT/logs/summary.log" 2>&1

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
assert len(rows) == 64, (len(rows), counts)
assert counts == Counter({method: 32 for method in expected}), counts
assert len(pairs) == 32
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
print("validated 32 strict Full/QK-balanced LongBench pairs")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
echo "ALL_COMPLETE"
