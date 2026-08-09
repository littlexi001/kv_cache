#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
REFERENCE=$ROOT/results/20260727_qkbalanced_longbench_official75k_full_8gpu
PHYSICAL=$ROOT/results/20260727_qk_variable_physical_128k_4gpu
ADDITIVITY=$ROOT/results/20260727_qkbalanced_additivity_closure_32k
PROGRESSIVE=$ROOT/results/20260727_progressive_variablebit_pipeline_3090
OAS_DEV=$ROOT/results/20260727_qkbalanced_qscale_oas_dev_m20_5gpu
MULTIGPU_PREFILL=$ROOT/results/20260727_qk_multigpu_resident_prefill_4gpu
QUANT_PREFILL=$ROOT/results/20260727_quantized_offloaded_prefill_gpu7
RUN_ROOT=$ROOT/results/20260727_qkbalanced_allocation_scale_factorial_offset100_m20_5gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS=countcap_fullprompt_qkbalanced_fixed4421_packed_direct,countcap_fullprompt_qkbalanced_qscale_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_packed_direct,countcap_fullprompt_qkbalanced_qscale_oas_packed_direct,countcap_fullprompt_qkbalanced_fixed4421_qscale_oas_packed_direct

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

while [[ ! -e "$PHYSICAL/ALL_COMPLETE" \
  || ! -e "$ADDITIVITY/ALL_COMPLETE" \
  || ! -e "$PROGRESSIVE/ALL_COMPLETE" \
  || ! -e "$OAS_DEV/ALL_COMPLETE" \
  || ! -e "$MULTIGPU_PREFILL/TERMINAL" \
  || ! -e "$QUANT_PREFILL/TERMINAL" ]]; do
  if [[ ! -e "$PHYSICAL/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_variable_physical_128k_4gpu_20260727.sh$' >/dev/null; then
    echo "128K physical prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$ADDITIVITY/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_additivity_after_norm_2gpu_20260727.sh$' >/dev/null; then
    echo "additivity prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$PROGRESSIVE/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_progressive_variablebit_pipeline_after_additivity_1gpu_20260727.sh$' >/dev/null; then
    echo "progressive CUDA prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$OAS_DEV/ALL_COMPLETE" ]] \
    && ! pgrep -f '^bash scripts/launch_qkbalanced_qscale_oas_dev_m20_5gpu_20260727.sh$' >/dev/null; then
    echo "OAS development prerequisite exited without ALL_COMPLETE" >&2
    exit 1
  fi
  if [[ ! -e "$MULTIGPU_PREFILL/TERMINAL" ]] \
    && ! pgrep -f '^bash scripts/launch_qk_multigpu_resident_prefill_after_oas_4gpu_20260727.sh$' >/dev/null; then
    echo "multi-GPU resident-prefill experiment exited without TERMINAL" >&2
    exit 1
  fi
  if [[ ! -e "$QUANT_PREFILL/TERMINAL" ]] \
    && ! pgrep -f '^bash scripts/launch_quantized_offloaded_prefill_probe_gpu7_20260727.sh$' >/dev/null; then
    echo "quantized offloaded-prefill experiment exited without TERMINAL" >&2
    exit 1
  fi
  sleep 60
done

gpus=(0 1 2 3 7)
pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --sample_offset_per_task 100 \
    --max_samples_per_task 20 \
    --num_shards 5 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
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
  echo "one or more holdout factorial shards failed; valid rows remain" >&2
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
assert len(rows) == 1600, (len(rows), counts)
assert counts == Counter({method: 320 for method in methods}), counts
assert len(pairs) == 320
assert all(value == methods for value in pairs.values())
assert len({row["task"] for row in rows}) == 16
print("validated 320 strict offset-100 five-cell factorial samples")
PY

"$PYTHON" src/analyze_qkbalanced_factorial_20260727.py \
  --reference_glob "$REFERENCE/shard[0-9]*/sample_results.csv" \
  --factorial_glob "$RUN_ROOT/shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/analysis" \
  --include_qscale_oas \
  >"$LOG_ROOT/analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
