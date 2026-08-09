#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
RUN_ROOT=$ROOT/results/20260723_countcap_direct_8k16k_4gpu
LOG_ROOT=$RUN_ROOT/logs
GPUS=(0 1 2 3)
LENGTHS=(8192 8192 16000 16000)
HORIZONS=(32 64 32 64)
REPEATS=3
SAMPLE_OFFSET=115
METHODS=full_kv,countcap_fullprompt_keypca,countcap_fullprompt_keypca_direct

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

research_gpus_idle() {
  local gpu
  for gpu in "${GPUS[@]}"; do
    if [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; then
      return 1
    fi
  done
  return 0
}

log "queued: waiting only for research GPUs 0-3; GPUs 4-7 are never used"
while ! research_gpus_idle; do
  sleep 30
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

log "starting Full vs exact-rerank 2% vs sampled-threshold direct attention"
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
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  log "one or more workers failed; preserving completed CSV files"
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" "$REPEATS" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
repeats = int(sys.argv[2])
methods = {
    "full_kv",
    "countcap_fullprompt_keypca",
    "countcap_fullprompt_keypca_direct",
}
for length, horizon in ((8192, 32), (8192, 64), (16000, 32), (16000, 64)):
    for repeat in range(1, repeats + 1):
        path = (
            root
            / f"length{length}"
            / f"g{horizon}"
            / f"repeat{repeat}"
            / "sample_results.csv"
        )
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3, (path, len(rows))
        assert {row["method"] for row in rows} == methods
        assert all(row["executed_path"] == row["method"] for row in rows)
        assert max(int(row["prompt_tokens"]) for row in rows) <= length
print("validated 4 cases x 3 repeats x 3 methods")
PY

"$PYTHON" src/analyze_countcap_direct_8k16k_20260723.py \
  --run_root "$RUN_ROOT" > "$LOG_ROOT/analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/summary.json"
