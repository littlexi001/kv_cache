#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
PREREQUISITE=$ROOT/results/20260727_qkbalanced_allocation_scale_factorial_offset100_m20_5gpu
RUN_ROOT=$ROOT/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=full_kv,countcap_fullprompt_qkbalanced_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_packed_direct,countcap_fullprompt_qkbalanced_qscale_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_packed_direct,countcap_fullprompt_qkbalanced_qscale_oas_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]]; do
  if ! pgrep -f '^bash scripts/launch_qkbalanced_factorial_holdout_offset100_5gpu_20260727.sh$' >/dev/null; then
    echo "holdout factorial exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHODS" \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 7500 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override 8 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 7
assert len({row["method"] for row in rows}) == 7
assert all(int(row["suffix_tokens"]) == 8 for row in rows)
assert all(int(row["prompt_tokens"]) <= 7512 for row in rows)
print("official-middle seven-way smoke passed")
PY

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,7}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  echo "QKSIEVE_GPUS must contain exactly five comma-separated GPU ids" >&2
  exit 2
fi
pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 0 \
    --num_shards 5 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more official-middle shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" <<'PY'
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("shard[0-9]*/sample_results.csv")):
    with path.open(encoding="utf-8", newline="") as handle:
        rows.extend(csv.DictReader(handle))
methods = {
    "full_kv",
    "countcap_fullprompt_qkbalanced_packed_direct",
    "countcap_fullprompt_qkbalanced_fixed4421_packed_direct",
    "countcap_fullprompt_qkbalanced_qscale_packed_direct",
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_packed_direct",
    "countcap_fullprompt_qkbalanced_qscale_oas_packed_direct",
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct",
}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])
assert len(rows) == 26250, (len(rows), counts)
assert counts == Counter({method: 3750 for method in methods}), counts
assert len(pairs) == 3750
assert all(value == methods for value in pairs.values())
assert len({row["task"] for row in rows}) == 16
assert max(int(row["prompt_tokens"]) for row in rows) <= 7512
assert {int(row["suffix_tokens"]) for row in rows} == {8}
print("validated 3750 strict official-middle seven-way samples")
PY

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" src/analyze_qkbalanced_factorial_20260727.py \
  --reference_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --factorial_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/analysis" \
  --include_qscale_oas \
  >"$LOG_ROOT/analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
