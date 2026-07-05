#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
LM_EVAL_DIR="${LM_EVAL_DIR:-/home/fdong/lm-evaluation-harness}"

source /home/fdong/miniconda3/bin/activate moe
cd "$LM_EVAL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/fdong/ymluo/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/fdong/ymluo/hf_cache/datasets}"

STAMP="${STAMP:-20260705_standard_8b}"
MODEL="${MODEL:-/home/fdong/qwen/LlaMa-3.1-8B}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/outputs/standard_8b_lm_eval_${STAMP}}"
LOG_ROOT="$PROJECT_DIR/outputs/logs"
BATCH_SIZE="${BATCH_SIZE:-auto}"
LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

# Common public leader-board style tasks. Few-shot values follow commonly used
# lm-eval defaults for these model sanity benchmarks.
TASK_SPECS="${TASK_SPECS:-mmlu:5 hellaswag:10 arc_challenge:25 winogrande:5 truthfulqa_mc2:0 gsm8k:5}"

STATUS="$OUT_ROOT/run_status.csv"
echo "task,num_fewshot,status,log,output_dir" > "$STATUS"

for spec in $TASK_SPECS; do
  task="${spec%%:*}"
  fewshot="${spec##*:}"
  task_out="$OUT_ROOT/$task"
  log="$LOG_ROOT/standard_8b_lm_eval_${task}_${fewshot}shot_${STAMP}.log"
  mkdir -p "$task_out"
  set +e
  python -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,dtype=float16,trust_remote_code=True" \
    --tasks "$task" \
    --num_fewshot "$fewshot" \
    --batch_size "$BATCH_SIZE" \
    --device cuda:0 \
    --output_path "$task_out" \
    "${LIMIT_ARG[@]}" \
    2>&1 | tee "$log"
  code=${PIPESTATUS[0]}
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "$task,$fewshot,OK,$log,$task_out" >> "$STATUS"
  else
    echo "$task,$fewshot,FAILED,$log,$task_out" >> "$STATUS"
    if [[ "${CONTINUE_ON_ERROR:-1}" != "1" ]]; then
      exit "$code"
    fi
  fi
done

python - "$OUT_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows: list[dict[str, str]] = []
for path in sorted(root.glob("**/*.json")):
    if path.name.startswith("samples_"):
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    results = data.get("results")
    if not isinstance(results, dict):
        continue
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not key.endswith("_stderr"):
                rows.append(
                    {
                        "task": str(task),
                        "metric": str(key),
                        "value": f"{float(value):.8g}",
                        "source": str(path),
                    }
                )
if rows:
    with (root / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "metric", "value", "source"])
        writer.writeheader()
        writer.writerows(rows)
print(root / "metrics_summary.csv")
PY

echo "output $OUT_ROOT"
