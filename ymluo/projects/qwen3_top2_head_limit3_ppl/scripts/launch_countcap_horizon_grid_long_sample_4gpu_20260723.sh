#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
DATA=/home/fdong/ymluo/external/KVCache-Factory/data/LongBench
PARENT_RUN=$ROOT/results/20260723_countcap_horizon_grid_4gpu
RUN_ROOT=$ROOT/results/20260723_countcap_horizon_grid_long_sample_4gpu
LOG_ROOT=$RUN_ROOT/logs
LENGTHS=(8192 16000 24576 32768)
HORIZONS=(8 32 64)
SAMPLE_OFFSET=115

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LOG_ROOT"
cd "$ROOT"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG_ROOT/launcher.log"
}

log "queued: waiting for the first horizon grid"
while [[ ! -f "$PARENT_RUN/ALL_COMPLETE" ]]; do
  sleep 60
done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')" ]]; do
  sleep 30
done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

log "starting corrected grid with GovReport source row 115 (reported length 40508)"
for gpu in 0 1 2 3; do
  length=${LENGTHS[$gpu]}
  (
    for horizon in "${HORIZONS[@]}"; do
      out="$RUN_ROOT/length${length}/g${horizon}"
      mkdir -p "$out"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u \
        src/run_sample_calibrated_longbench_20260717.py \
        --model_name_or_path "$MODEL" \
        --longbench_data_dir "$DATA" \
        --output_dir "$out" \
        --tasks gov_report \
        --methods full_kv,countcap_fullprompt_keypca \
        --sample_offset_per_task "$SAMPLE_OFFSET" \
        --max_samples_per_task 1 \
        --num_shards 1 --shard_index 0 \
        --max_prompt_tokens "$length" \
        --max_context_tokens 0 \
        --max_new_tokens_override "$horizon" \
        --prefill_chunk_tokens 2048 \
        --prompt_wrapper llama3 \
        --dtype float16 --device cuda --device_map auto \
        > "$LOG_ROOT/length${length}_g${horizon}.log" 2>&1
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
  log "one or more grid workers failed; preserving completed rows"
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" "${LENGTHS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
for length in map(int, sys.argv[2:]):
    for horizon in (8, 32, 64):
        path = root / f"length{length}" / f"g{horizon}" / "sample_results.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2, (path, len(rows))
        assert {row["method"] for row in rows} == {
            "full_kv",
            "countcap_fullprompt_keypca",
        }
        prompt_tokens = {int(row["prompt_tokens"]) for row in rows}
        assert len(prompt_tokens) == 1
        assert next(iter(prompt_tokens)) <= length
print("validated 4 prompt lengths x 3 horizons x 2 methods")
PY

"$PYTHON" src/analyze_countcap_horizon_grid_20260723.py \
  --run_root "$RUN_ROOT" > "$LOG_ROOT/analysis.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
log "complete: $RUN_ROOT/horizon_grid.json"
