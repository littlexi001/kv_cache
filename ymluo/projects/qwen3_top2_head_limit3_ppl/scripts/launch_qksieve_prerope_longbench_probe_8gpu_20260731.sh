#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260731_qksieve_prerope_longbench_probe_qwen3_4b_m6_32k_8gpu}"
TASKS="${TASKS:-narrativeqa,qasper,hotpotqa,musique,qmsum,gov_report,passage_retrieval_en,lcc}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-6}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-32000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GPUS_CSV="${QKSIEVE_GPUS:-0,1,2,3,4,5,6,7}"
BASE_METHOD="qksieve_fullprompt_fixed410_fulltopk"
PREROPE_METHOD="qksieve_fullprompt_fixed410_post2xprererank_l00to08_fulltopk"
METHODS="full_kv,$BASE_METHOD,$PREROPE_METHOD"
LOG_ROOT="$RUN_ROOT/logs"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

if [[ ! -f "$MODEL/config.json" ]]; then
  echo "model is incomplete: $MODEL" >&2
  exit 2
fi
if [[ ! -d "$DATA" ]]; then
  echo "LongBench data directory is missing: $DATA" >&2
  exit 2
fi

IFS=',' read -r -a gpus <<< "$GPUS_CSV"
if [[ "${#gpus[@]}" -ne 8 ]]; then
  echo "QKSIEVE_GPUS must contain exactly eight GPU ids" >&2
  exit 2
fi
declare -A seen=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]] || [[ -n "${seen[$gpu]+x}" ]]; then
    echo "invalid or duplicate GPU id: $gpu" >&2
    exit 2
  fi
  seen[$gpu]=1
done

CUDA_VISIBLE_DEVICES="${gpus[0]}" "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHODS" \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 8192 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override 8 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper qwen3 \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" "$BASE_METHOD" "$PREROPE_METHOD" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
expected = {"full_kv", sys.argv[2], sys.argv[3]}
assert len(rows) == 3, len(rows)
assert {row["method"] for row in rows} == expected
for row in rows:
    if row["method"] == "full_kv":
        continue
    assert float(row["configured_index_bits_per_token"]) == 112.0
    assert int(row["configured_attention_tokens"]) > 0
print("strict fixed410/pre-RoPE LongBench smoke passed")
PY

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${gpus[$shard]}" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task "$SAMPLES_PER_TASK" \
    --num_shards 8 \
    --shard_index "$shard" \
    --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override "$MAX_NEW_TOKENS" \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper qwen3 \
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
  echo "one or more shards failed; valid CSV rows were preserved" >&2
  exit 1
fi

"$PYTHON" src/summarize_qksieve_prerope_longbench_probe_20260731.py \
  --run_root "$RUN_ROOT" \
  --tasks "$TASKS" \
  --samples_per_task "$SAMPLES_PER_TASK" \
  >"$LOG_ROOT/summarize.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
echo "ALL_COMPLETE $RUN_ROOT"
