#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
WORK_ROOT="${WORK_ROOT:-$PROJECT_ROOT/portable_countcap_workspace}"
RUN_ID="${RUN_ID:-countcap_qwen3_4b_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$WORK_ROOT/runs/$RUN_ID"
PROTOCOL="${PROTOCOL:-paper}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RULER_LENGTHS="${RULER_LENGTHS:-65536,131072}"
SEED="${SEED:-42}"

if [[ "$PROTOCOL" == "quick" ]]; then
  LONG_SAMPLES="${LONG_SAMPLES:-5}"
  RULER_SAMPLES="${RULER_SAMPLES:-1}"
  LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"
else
  # Zero means all official LongBench examples (3750 in the 16 English tasks).
  LONG_SAMPLES="${LONG_SAMPLES:-0}"
  RULER_SAMPLES="${RULER_SAMPLES:-5}"
  # Zero preserves each LongBench task's official generation length.
  LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-0}"
fi

MODEL_DIR="$WORK_ROOT/models/Qwen3-4B-Instruct-2507"
LONG_DATA="$WORK_ROOT/data/longbench/data"
RULER_DATA="$WORK_ROOT/data/ruler/qwen3_4b_${RULER_LENGTHS//,/_}_m${RULER_SAMPLES}_seed${SEED}.jsonl"
UTILITY="$PROJECT_ROOT/scripts/portable_countcap_benchmark.py"
LONG_TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
RULER_TASKS=niah_single_1,niah_multikey_1,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_cache}"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
mkdir -p "$RUN_ROOT/logs"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN_ROOT/logs/launcher.log"
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
    log "one or more benchmark shards failed; inspect $RUN_ROOT/logs"
    return 1
  fi
}

if [[ "$INSTALL_DEPS" == "1" ]]; then
  log "installing Python dependencies (PyTorch/CUDA must already be installed)"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install \
    "transformers==4.53.0" \
    "accelerate==0.33.0" \
    "huggingface_hub==0.33.0" \
    "datasets>=2.19,<4" \
    ninja packaging sentencepiece protobuf pandas pyarrow requests tqdm \
    rouge rouge-score jieba fuzzywuzzy python-Levenshtein nltk wonderwords scipy
fi

"$PYTHON" "$UTILITY" doctor \
  --project_root "$PROJECT_ROOT" \
  --expected_gpus 8 \
  --output "$RUN_ROOT/environment.json"

log "downloading model/data and preparing frozen RULER examples"
prepare_args=(
  prepare
  --project_root "$PROJECT_ROOT"
  --work_root "$WORK_ROOT"
  --ruler_lengths "$RULER_LENGTHS"
  --ruler_samples "$RULER_SAMPLES"
  --seed "$SEED"
)
if [[ "$INSTALL_DEPS" == "1" ]]; then
  prepare_args+=(--install_lm_eval)
fi
"$PYTHON" -u "$UTILITY" "${prepare_args[@]}" 2>&1 \
  | tee "$RUN_ROOT/logs/prepare.log"

"$PYTHON" "$UTILITY" doctor \
  --project_root "$PROJECT_ROOT" \
  --expected_gpus 8 \
  --output "$RUN_ROOT/environment.json"

log "warming and compiling CountCap CUDA kernels on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  "$PROJECT_ROOT/src/run_sample_calibrated_longbench_20260717.py" \
  --model_name_or_path "$MODEL_DIR" \
  --longbench_data_dir "$LONG_DATA" \
  --output_dir "$RUN_ROOT/longbench_smoke" \
  --tasks narrativeqa \
  --methods full_kv,countcap \
  --max_samples_per_task 1 \
  --max_context_tokens 7500 \
  --max_new_tokens_override 16 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper qwen3 \
  --dtype float16 --device cuda --device_map auto \
  > "$RUN_ROOT/logs/smoke.log" 2>&1

log "starting LongBench: protocol=$PROTOCOL samples_per_task=$LONG_SAMPLES"
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON" -u \
    "$PROJECT_ROOT/src/run_sample_calibrated_longbench_20260717.py" \
    --model_name_or_path "$MODEL_DIR" \
    --longbench_data_dir "$LONG_DATA" \
    --output_dir "$RUN_ROOT/longbench_shard${shard}" \
    --tasks "$LONG_TASKS" \
    --methods full_kv,countcap \
    --max_samples_per_task "$LONG_SAMPLES" \
    --num_shards 8 --shard_index "$shard" \
    --max_context_tokens 7500 \
    --max_new_tokens_override "$LONG_MAX_NEW_TOKENS" \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper qwen3 \
    --dtype float16 --device cuda --device_map auto \
    > "$RUN_ROOT/logs/longbench_shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
wait_group
touch "$RUN_ROOT/LONGBENCH_COMPLETE"

min_gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if [[ "${RULER_GPUS_PER_JOB:-auto}" == "auto" ]]; then
  if (( min_gpu_mib >= 45000 )); then
    RULER_GPUS_PER_JOB=1
  else
    RULER_GPUS_PER_JOB=2
  fi
fi
if [[ "$RULER_GPUS_PER_JOB" == "1" ]]; then
  RULER_JOBS=8
  RULER_DEVICE_MAP=auto
else
  RULER_GPUS_PER_JOB=2
  RULER_JOBS=4
  RULER_DEVICE_MAP=balanced
fi

log "starting RULER: samples_per_task_length=$RULER_SAMPLES jobs=$RULER_JOBS gpus_per_job=$RULER_GPUS_PER_JOB"
for ((shard = 0; shard < RULER_JOBS; shard++)); do
  if [[ "$RULER_GPUS_PER_JOB" == "1" ]]; then
    visible="$shard"
  else
    first=$((2 * shard))
    visible="$first,$((first + 1))"
  fi
  CUDA_VISIBLE_DEVICES="$visible" "$PYTHON" -u \
    "$PROJECT_ROOT/src/run_sample_calibrated_ruler_20260717.py" \
    --model_name_or_path "$MODEL_DIR" \
    --examples_jsonl "$RULER_DATA" \
    --output_dir "$RUN_ROOT/ruler_shard${shard}" \
    --methods full_kv,countcap \
    --ruler_tasks "$RULER_TASKS" \
    --ruler_lengths "$RULER_LENGTHS" \
    --max_samples_per_task "$RULER_SAMPLES" \
    --num_shards "$RULER_JOBS" --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper none \
    --dtype float16 --device cuda --device_map "$RULER_DEVICE_MAP" \
    > "$RUN_ROOT/logs/ruler_shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done
wait_group
touch "$RUN_ROOT/RULER_COMPLETE"

log "validating pairs and generating final report"
"$PYTHON" "$UTILITY" finalize \
  --project_root "$PROJECT_ROOT" \
  --longbench_glob "$RUN_ROOT/longbench_shard*/sample_results.csv" \
  --ruler_glob "$RUN_ROOT/ruler_shard*/sample_results.csv" \
  --output_dir "$RUN_ROOT/final" \
  > "$RUN_ROOT/logs/finalize.log" 2>&1

cp "$RUN_ROOT/environment.json" "$RUN_ROOT/final/environment.json"
cp "$WORK_ROOT/assets.json" "$RUN_ROOT/final/assets.json"

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/final/RESULTS.md"
