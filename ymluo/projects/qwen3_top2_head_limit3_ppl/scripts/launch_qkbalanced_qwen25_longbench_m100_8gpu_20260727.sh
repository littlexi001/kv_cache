#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen2.5-7B-Instruct
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
REFERENCE=$ROOT/results/20260726_countcap_qwen25_7b_longbench_m100_prompt7500
PREREQUISITE=$ROOT/results/20260727_qkbalanced_ruler_4k128k_8gpu
RUN_ROOT=$ROOT/results/20260727_qkbalanced_qwen25_7b_longbench_m100_8gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=countcap_fullprompt_qkbalanced_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_packed_direct,countcap_fullprompt_qkbalanced_qscale_oas_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
  if ! pgrep -f '^bash scripts/launch_qkbalanced_ruler_4k128k_8gpu_20260727.sh$' >/dev/null; then
    echo "RULER prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

if [[ ! -e "$REFERENCE/ALL_COMPLETE" ]]; then
  echo "missing completed Qwen2.5 Full-KV reference" >&2
  exit 2
fi

run_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 100 \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_prompt_tokens 7500 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper qwen3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/shard${shard}.log" 2>&1
}

pids=()
(run_shard 0 0; run_shard 0 5) & pids+=("$!")
(run_shard 1 1; run_shard 1 6) & pids+=("$!")
(run_shard 2 2; run_shard 2 7) & pids+=("$!")
run_shard 3 3 & pids+=("$!")
run_shard 7 4 & pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more Qwen2.5 QK-balanced shards failed" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/merged_input"
"$PYTHON" - "$REFERENCE" "$RUN_ROOT" "$METHODS" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

reference = Path(sys.argv[1])
run_root = Path(sys.argv[2])
sparse_methods = set(sys.argv[3].split(","))

full_rows = []
for path in sorted(reference.glob("shard[0-9]*/sample_results.csv")):
    with path.open(encoding="utf-8", newline="") as handle:
        full_rows.extend(
            row for row in csv.DictReader(handle) if row["method"] == "full_kv"
        )

sparse_rows = []
for path in sorted(run_root.glob("shard[0-9]*/sample_results.csv")):
    with path.open(encoding="utf-8", newline="") as handle:
        sparse_rows.extend(csv.DictReader(handle))

rows = full_rows + sparse_rows
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

expected = {"full_kv", *sparse_methods}
assert counts == Counter({method: 1600 for method in expected}), counts
assert len(rows) == 1600 * len(expected)
assert len(pairs) == 1600
assert all(methods == expected for methods in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(float(row["prompt_tokens"])) for row in rows) <= 7500

output = run_root / "merged_input" / "sample_results.csv"
fieldnames = list(full_rows[0])
for row in sparse_rows:
    for field in row:
        if field not in fieldnames:
            fieldnames.append(field)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"validated and wrote 1600 strict {len(expected)}-way Qwen2.5 samples")
PY

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/merged_input/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" src/analyze_qkbalanced_longbench_paired_20260727.py \
  --input_csv "$RUN_ROOT/merged/sample_results.csv" \
  --output_dir "$RUN_ROOT/paired_analysis" \
  >"$LOG_ROOT/paired_analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
