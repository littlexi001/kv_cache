#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
PRIMARY_PID="${PRIMARY_PID:-0}"
LONG_LAUNCHER="$ROOT/scripts/launch_qksieve_fulltopk_longbench_5gpu_20260728.sh"
MULTIMODEL_LAUNCHER="$ROOT/scripts/launch_qksieve_three_model_longbench_20260728.sh"
RULER_LAUNCHER="$ROOT/scripts/launch_qksieve_fulltopk_ruler_6gpu_20260728.sh"
SUMMARY_ROOT="$ROOT/results/20260728_qksieve_three_model_longbench"
QUEUE_ROOT="$ROOT/results/20260728_qksieve_submission_queue_6gpu"

LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
MISTRAL_MODEL="${MISTRAL_MODEL:-/home/fdong/models/Mistral-7B-Instruct-v0.3}"
MISTRAL_REPO="${MISTRAL_REPO:-mistralai/Mistral-7B-Instruct-v0.3}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_DISABLE_XET=1
mkdir -p "$QUEUE_ROOT/logs" "$SUMMARY_ROOT"

if [[ "$PRIMARY_PID" =~ ^[1-9][0-9]*$ ]]; then
  while kill -0 "$PRIMARY_PID" 2>/dev/null; do
    sleep 60
  done
fi

model_complete() {
  "$PYTHON" - "$1" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
indices = list(root.glob("*.safetensors.index.json"))
if indices:
    index = json.loads(indices[0].read_text(encoding="utf-8"))
    weights = {root / name for name in index["weight_map"].values()}
else:
    weights = set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin"))
if not weights or any(not path.is_file() or path.stat().st_size == 0 for path in weights):
    raise SystemExit(1)
PY
}

for attempt in 1 2 3; do
  if model_complete "$MISTRAL_MODEL"; then
    break
  fi
  echo "Mistral download attempt $attempt"
  if HF_ENDPOINT=https://huggingface.co \
    huggingface-cli download "$MISTRAL_REPO" \
      --local-dir "$MISTRAL_MODEL"; then
    model_complete "$MISTRAL_MODEL" && break
  fi
  sleep $((attempt * 60))
done
model_complete "$MISTRAL_MODEL"

run_longbench() {
  local tag="$1"
  local model="$2"
  local wrapper="$3"
  local output="$SUMMARY_ROOT/$tag"
  if [[ -e "$output/ALL_COMPLETE" ]]; then
    echo "[skip] $tag LongBench is complete"
    return
  fi
  MODEL="$model" \
  MODEL_TAG="$tag" \
  PROMPT_WRAPPER="$wrapper" \
  RUN_ROOT="$output" \
  QKSIEVE_GPUS=0,1,2,3,4 \
    bash "$LONG_LAUNCHER"
}

run_longbench llama31_8b "$LLAMA_MODEL" llama3 \
  >"$QUEUE_ROOT/logs/continuation_llama31_8b.log" 2>&1
run_longbench qwen3_4b "$QWEN_MODEL" qwen3 \
  >"$QUEUE_ROOT/logs/continuation_qwen3_4b.log" 2>&1
run_longbench mistral_7b "$MISTRAL_MODEL" tokenizer_chat \
  >"$QUEUE_ROOT/logs/continuation_mistral_7b.log" 2>&1

QKSIEVE_GPUS=0,1,2,3,4 \
  bash "$MULTIMODEL_LAUNCHER" \
  >"$QUEUE_ROOT/logs/continuation_multimodel_summary.log" 2>&1
touch "$QUEUE_ROOT/LONGBENCH_COMPLETE"

bash "$RULER_LAUNCHER" \
  >"$QUEUE_ROOT/logs/continuation_ruler.log" 2>&1
touch "$QUEUE_ROOT/ALL_COMPLETE"
