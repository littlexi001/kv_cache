#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
GPU="${QKSIEVE_GPU:-0}"
VALIDATION_MATRIX="${VALIDATION_MATRIX:-$ROOT/results/20260728_qksieve_qfused_correctness/validation_matrix.json}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/20260728_qksieve_qfused_longbench_smoke}"
TASKS="${QKSIEVE_SMOKE_TASKS:-narrativeqa,hotpotqa,gov_report,repobench-p}"
METHODS=full_kv,qksieve_fullprompt_auto_plain_fulltopk,qksieve_fullprompt_auto_plain_qfused_fulltopk

if [[ ! "$GPU" =~ ^[0-5]$ ]]; then
  echo "QKSIEVE_GPU must be one of physical GPUs 0-5" >&2
  exit 2
fi

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$OUT_ROOT"
cd "$ROOT"

"$PYTHON" - "$VALIDATION_MATRIX" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(
        "missing qfused validation matrix; run "
        "scripts/run_qksieve_qfused_correctness_20260728.sh first"
    )
report = json.loads(path.read_text(encoding="utf-8"))
if report.get("all_passed") is not True:
    raise SystemExit("qfused validation matrix did not pass")
print("qfused numerical/latency matrix passed")
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$OUT_ROOT" \
  --tasks "$TASKS" \
  --methods "$METHODS" \
  --max_samples_per_task "${QKSIEVE_SMOKE_SAMPLES_PER_TASK:-2}" \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 7500 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override "${QKSIEVE_SMOKE_MAX_NEW_TOKENS:-64}" \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --qk_metric_query_shrinkage 0.75 \
  --collect_attention_stats \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  2>&1 | tee "$OUT_ROOT/run.log"

"$PYTHON" \
  src/analyze_qksieve_qfused_longbench_smoke_20260728.py \
  --results "$OUT_ROOT/sample_results.csv" \
  --validation_matrix "$VALIDATION_MATRIX" \
  --output "$OUT_ROOT/smoke_report.json" \
  --min_prediction_match "${QKSIEVE_SMOKE_MIN_PREDICTION_MATCH:-0.875}" \
  --max_mean_score_delta "${QKSIEVE_SMOKE_MAX_MEAN_SCORE_DELTA:-0.01}" \
  2>&1 | tee "$OUT_ROOT/analyze.log"
