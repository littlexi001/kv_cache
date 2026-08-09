#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/Qwen3-4B-Instruct
LONG_DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RULER_DATA=$ROOT/data/ruler_generated/llama31_8b_64k128k_m5_seed42.jsonl
RUN_ID=20260722_countcap_qwen4b
LOG_ROOT=$ROOT/results/${RUN_ID}_logs
WAIT_PIDS=(3547904 3547906 3547907 3547908)
LONG_TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
RULER_TASKS=niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot

export PATH=/home/fdong/miniconda3/envs/moe/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

while true; do
  running=0
  for pid in "${WAIT_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=1
    fi
  done
  if [[ "$running" -eq 0 ]]; then
    break
  fi
  log "waiting for existing GPU jobs: ${WAIT_PIDS[*]}"
  sleep 60
done

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
  log "all GPUs are still occupied by compute processes; waiting"
  sleep 60
done

log "validating CountCap runner"
"$PYTHON" -m py_compile \
  src/run_sample_calibrated_longbench_20260717.py \
  src/run_sample_calibrated_ruler_20260717.py \
  src/summarize_countcap_benchmark_20260722.py

log "starting LongBench CountCap smoke"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$LONG_DATA" \
  --output_dir "results/${RUN_ID}_longbench_smoke" \
  --tasks narrativeqa \
  --methods full_kv,countcap \
  --max_samples_per_task 1 \
  --max_context_tokens 7500 \
  --max_new_tokens_override 32 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper qwen3 \
  --dtype float16 --device cuda --device_map auto \
  > "$LOG_ROOT/longbench_smoke.log" 2>&1

log "LongBench smoke passed; starting 16-task m5 run on five GPUs"
long_pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$LONG_DATA" \
    --output_dir "results/${RUN_ID}_longbench_m5_shard${shard}" \
    --tasks "$LONG_TASKS" \
    --methods full_kv,countcap \
    --max_samples_per_task 5 \
    --num_shards 5 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override 64 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper qwen3 \
    --dtype float16 --device cuda --device_map auto \
    > "$LOG_ROOT/longbench_shard${shard}.log" 2>&1 &
  long_pids+=("$!")
done
wait "${long_pids[@]}"

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind longbench \
  --input_glob "results/${RUN_ID}_longbench_m5_shard*/sample_results.csv" \
  --output_dir "results/${RUN_ID}_longbench_m5_merged" \
  > "$LOG_ROOT/longbench_summary.log" 2>&1
touch "results/${RUN_ID}_LONGBENCH_COMPLETE"
log "LongBench complete; starting RULER smoke on GPUs 0,1"

CUDA_VISIBLE_DEVICES=0,1 "$PYTHON" -u src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "$MODEL" \
  --examples_jsonl "$RULER_DATA" \
  --output_dir "results/${RUN_ID}_ruler_smoke" \
  --methods full_kv,countcap \
  --ruler_tasks niah_single_1 \
  --ruler_lengths 65536,131072 \
  --max_samples_per_task 1 \
  --max_new_tokens_override 32 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper none \
  --dtype float16 --device cuda --device_map balanced \
  > "$LOG_ROOT/ruler_smoke.log" 2>&1

log "RULER smoke passed; starting nine-task 64K/128K m1 run on four GPU pairs"
ruler_pids=()
for shard in 0 1 2 3; do
  first=$((2 * shard))
  second=$((first + 1))
  CUDA_VISIBLE_DEVICES="$first,$second" "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$RULER_DATA" \
    --output_dir "results/${RUN_ID}_ruler_64k128k_m1_shard${shard}" \
    --methods full_kv,countcap \
    --ruler_tasks "$RULER_TASKS" \
    --ruler_lengths 65536,131072 \
    --max_samples_per_task 1 \
    --num_shards 4 --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --dtype float16 --device cuda --device_map balanced \
    > "$LOG_ROOT/ruler_shard${shard}.log" 2>&1 &
  ruler_pids+=("$!")
done
wait "${ruler_pids[@]}"

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "results/${RUN_ID}_ruler_64k128k_m1_shard*/sample_results.csv" \
  --output_dir "results/${RUN_ID}_ruler_64k128k_m1_merged" \
  > "$LOG_ROOT/ruler_summary.log" 2>&1
touch "results/${RUN_ID}_RULER_COMPLETE"
log "all CountCap benchmark stages complete"
