#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
SHORT_DATA=$ROOT/data/ruler_generated/llama31_8b_4k32k_m10_seed42.jsonl
LONG_DATA=$ROOT/data/ruler_generated/llama31_8b_64k128k_m5_seed42.jsonl
PREREQUISITE=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
PROGRESSIVE=$ROOT/results/20260727_qk_progressive_refinement_32k
MATCHED_RATE=$ROOT/results/20260727_qk_matched_rate_all_dims_32k
NORM_CERTIFIED=$ROOT/results/20260727_qk_norm_certified_refinement_32k
ADDITIVITY=$ROOT/results/20260727_qkbalanced_additivity_closure_32k
FACTORIAL_HOLDOUT=$ROOT/results/20260727_qkbalanced_allocation_scale_factorial_offset100_m20_5gpu
OFFICIAL_MIDDLE=$ROOT/results/20260727_qkbalanced_longbench_official_middle_5way_5gpu
PARETO_128K_2GPU=$ROOT/results/20260727_qkbalanced_128k_2gpu_pareto
RUN_ROOT=$ROOT/results/20260727_qkbalanced_ruler_4k128k_8gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
METHODS=full_kv,countcap_fullprompt_qkbalanced_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_packed_direct,countcap_fullprompt_qkbalanced_qscale_oas_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PREREQUISITE/ALL_COMPLETE" \
  || ! -e "$PROGRESSIVE/ALL_COMPLETE" \
  || ! -e "$MATCHED_RATE/ALL_COMPLETE" \
  || ! -e "$NORM_CERTIFIED/ALL_COMPLETE" \
  || ! -e "$ADDITIVITY/ALL_COMPLETE" \
  || ! -e "$FACTORIAL_HOLDOUT/ALL_COMPLETE" \
  || ! -e "$OFFICIAL_MIDDLE/ALL_COMPLETE" \
  || ! -e "$PARETO_128K_2GPU/ALL_COMPLETE" ]]; do
  if [[ ! -e "$PREREQUISITE/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_variable_physical_128k_4gpu_20260727.sh$' >/dev/null; then
    echo "128K physical prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$PROGRESSIVE/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_progressive_refinement_after_physical_2gpu_20260727.sh$' >/dev/null; then
    echo "progressive-refinement prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$MATCHED_RATE/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_matched_rate_after_physical_2gpu_20260727.sh$' >/dev/null; then
    echo "matched-rate prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$NORM_CERTIFIED/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_norm_certified_after_matched_2gpu_20260727.sh$' >/dev/null; then
    echo "norm-certified prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$ADDITIVITY/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_additivity_after_norm_2gpu_20260727.sh$' >/dev/null; then
    echo "additivity prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$FACTORIAL_HOLDOUT/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_factorial_holdout_offset100_5gpu_20260727.sh$' >/dev/null; then
    echo "holdout factorial prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$OFFICIAL_MIDDLE/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_longbench_official_middle_5way_5gpu_20260727.sh$' >/dev/null; then
    echo "official-middle LongBench prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$PARETO_128K_2GPU/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_128k_2gpu_pareto_20260727.sh$' >/dev/null; then
    echo "128K two-GPU Pareto prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  sleep 60
done

if [[ ! -s "$SHORT_DATA" || ! -s "$LONG_DATA" ]]; then
  echo "missing frozen RULER data" >&2
  exit 2
fi

# One cheap smoke catches shared-runner regressions before the full queue.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "$MODEL" \
  --examples_jsonl "$SHORT_DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --methods "$METHODS" \
  --ruler_tasks niah_single_1 \
  --ruler_lengths 4096 \
  --max_samples_per_task 1 \
  --max_new_tokens_override 16 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper none \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

run_short_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$SHORT_DATA" \
    --output_dir "$RUN_ROOT/short_shard${shard}" \
    --methods "$METHODS" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 4096,8192,16384,32768 \
    --max_samples_per_task 10 \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/short_shard${shard}.log" 2>&1
}

short_pids=()
(run_short_shard 0 0; run_short_shard 0 5) & short_pids+=("$!")
(run_short_shard 1 1; run_short_shard 1 6) & short_pids+=("$!")
(run_short_shard 2 2; run_short_shard 2 7) & short_pids+=("$!")
run_short_shard 3 3 & short_pids+=("$!")
run_short_shard 7 4 & short_pids+=("$!")
failed=0
for pid in "${short_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more short RULER shards failed" >&2
  exit 1
fi

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$LONG_DATA" \
    --output_dir "$RUN_ROOT/long_shard${shard}" \
    --methods "$METHODS" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 65536,131072 \
    --max_samples_per_task 5 \
    --num_shards 2 \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"$LOG_ROOT/long_shard${shard}.log" 2>&1
done

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "$RUN_ROOT/*_shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" <<'PY'
import csv
import sys
from collections import Counter, defaultdict

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected = {
    "full_kv",
    "countcap_fullprompt_qkbalanced_packed_direct",
    "countcap_fullprompt_qkbalanced_fixed4421_packed_direct",
    "countcap_fullprompt_qkbalanced_qscale_oas_packed_direct",
    "countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct",
}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 2250, (len(rows), counts)
assert counts == Counter({method: 450 for method in expected}), counts
assert len(pairs) == 450
assert all(methods == expected for methods in pairs.values())
assert len({row["base_task"] for row in rows}) == 9
assert {int(row["requested_length"]) for row in rows} == {
    4096, 8192, 16384, 32768, 65536, 131072
}
print("validated 450 strict five-way QK-balanced RULER samples")
PY

touch "$RUN_ROOT/ALL_COMPLETE"
