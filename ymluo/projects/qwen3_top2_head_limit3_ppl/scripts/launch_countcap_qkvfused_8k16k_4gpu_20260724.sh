#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260724_countcap_qkvfused_8k16k_4gpu_retry1
LOG_ROOT=$RUN_ROOT/logs
GPUS=(0 1 2 3)
LENGTHS=(8192 8192 16000 16000)
HORIZONS=(32 64 32 64)
REPEATS=3
SAMPLE_OFFSET=115
METHODS=full_kv,countcap_fullprompt_keypca_direct_fused,countcap_fullprompt_keypca_direct_qkvfused

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST=8.6
mkdir -p "$LOG_ROOT"
cd "$ROOT"

for gpu in "${GPUS[@]}"; do
  if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
    echo "GPU $gpu is busy; refusing to interfere" >&2
    exit 1
  fi
done

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for slot in 0 1 2 3; do
  gpu=${GPUS[$slot]}
  length=${LENGTHS[$slot]}
  horizon=${HORIZONS[$slot]}
  (
    for repeat in $(seq 1 "$REPEATS"); do
      out="$RUN_ROOT/length${length}/g${horizon}/repeat${repeat}"
      mkdir -p "$out"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        src/run_sample_calibrated_longbench_20260717.py \
        --model_name_or_path "$MODEL" \
        --longbench_data_dir "$DATA" \
        --output_dir "$out" \
        --tasks gov_report \
        --methods "$METHODS" \
        --sample_offset_per_task "$SAMPLE_OFFSET" \
        --max_samples_per_task 1 \
        --num_shards 1 --shard_index 0 \
        --max_prompt_tokens "$length" \
        --max_context_tokens 0 \
        --max_new_tokens_override "$horizon" \
        --prefill_chunk_tokens 2048 \
        --prompt_wrapper llama3 \
        --dtype float16 --device cuda --device_map auto \
        > "$LOG_ROOT/length${length}_g${horizon}_repeat${repeat}.log" 2>&1
    done
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more workers failed; completed results were preserved" >&2
  exit 1
fi

touch "$RUN_ROOT/ALL_COMPLETE"
echo "$RUN_ROOT"
