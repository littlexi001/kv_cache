#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
DATA=$ROOT/data/ruler_generated/qwen3_4b_64k128k_m5_seed42.jsonl
RUN_ROOT=$ROOT/results/20260722_countcap_qwen4b_ruler_m5_7gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

wait_group() {
  local failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if [[ "$failed" -ne 0 ]]; then
    log "one or more shards failed; inspect $LOG_ROOT"
    return 1
  fi
}

if [[ ! -s "$DATA" ]]; then
  log "missing frozen Qwen RULER data: $DATA"
  exit 2
fi

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
  log "GPU compute processes already exist; refusing to mix timing runs"
  nvidia-smi
  exit 3
fi

log "running 64K CountCap smoke on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "$MODEL" \
  --examples_jsonl "$DATA" \
  --output_dir "$RUN_ROOT/smoke64k" \
  --methods full_kv,countcap \
  --ruler_tasks niah_single_1 \
  --ruler_lengths 65536 \
  --max_samples_per_task 1 \
  --max_new_tokens_override 16 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper none \
  --dtype float16 --device cuda --device_map auto \
  > "$LOG_ROOT/smoke64k.log" 2>&1

log "smoke passed; starting 64K m5 on GPUs 0-6 (seven single-GPU shards)"
for shard in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$DATA" \
    --output_dir "$RUN_ROOT/64k_shard${shard}" \
    --methods full_kv,countcap \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 65536 \
    --max_samples_per_task 5 \
    --num_shards 7 --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/64k_shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
wait_group
touch "$RUN_ROOT/RULER_64K_COMPLETE"

log "64K complete; starting 128K m5 on GPUs 0-6 (3+2+2 layout)"
VISIBLE_GROUPS=("0,1,2" "3,4" "5,6")
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="${VISIBLE_GROUPS[$shard]}" "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$DATA" \
    --output_dir "$RUN_ROOT/128k_shard${shard}" \
    --methods full_kv,countcap \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 131072 \
    --max_samples_per_task 5 \
    --num_shards 3 --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --dtype float16 --device cuda --device_map balanced \
    > "$LOG_ROOT/128k_shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
wait_group
touch "$RUN_ROOT/RULER_128K_COMPLETE"

log "all shards complete; merging Full KV and CountCap results"
"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "$RUN_ROOT/*k_shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  > "$LOG_ROOT/summary.log" 2>&1
touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/merged/summary.json"
