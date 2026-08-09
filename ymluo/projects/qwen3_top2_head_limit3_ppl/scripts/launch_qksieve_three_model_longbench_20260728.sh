#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
GPUS="${QKSIEVE_GPUS:-0,1,2,3,4}"
ALLOW_DOWNLOAD="${QKSIEVE_DOWNLOAD_MISSING_MODELS:-1}"
LAUNCHER="$ROOT/scripts/launch_qksieve_fulltopk_longbench_5gpu_20260728.sh"
SUMMARY_ROOT="$ROOT/results/20260728_qksieve_three_model_longbench"

LLAMA_MODEL="${LLAMA_MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
QWEN_MODEL="${QWEN_MODEL:-/home/fdong/models/Qwen3-4B-Instruct}"
THIRD_MODEL="${THIRD_MODEL:-/home/fdong/models/Mistral-7B-Instruct-v0.3}"
THIRD_REPO="${THIRD_REPO:-mistralai/Mistral-7B-Instruct-v0.3}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p "$SUMMARY_ROOT/logs"

model_complete() {
  "$PYTHON" - "$1" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not (root / "config.json").is_file():
    raise SystemExit(1)
index_paths = list(root.glob("*.safetensors.index.json"))
if index_paths:
    with index_paths[0].open(encoding="utf-8") as handle:
        index = json.load(handle)
    files = {root / name for name in index["weight_map"].values()}
else:
    files = set(root.glob("*.safetensors")) | set(root.glob("pytorch_model*.bin"))
if not files or any(not path.is_file() or path.stat().st_size == 0 for path in files):
    raise SystemExit(1)
PY
}

download_model() {
  local repo="$1"
  local destination="$2"
  if [[ "$ALLOW_DOWNLOAD" != "1" ]]; then
    echo "missing model $destination; set QKSIEVE_DOWNLOAD_MISSING_MODELS=1" >&2
    return 2
  fi
  "$PYTHON" - "$repo" "$destination" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN"),
)
PY
}

ensure_model() {
  local path="$1"
  local repo="${2:-}"
  if model_complete "$path"; then
    return
  fi
  if [[ -z "$repo" ]]; then
    echo "model is incomplete and has no public fallback repo: $path" >&2
    exit 2
  fi
  download_model "$repo" "$path"
  model_complete "$path"
}

run_model() {
  local tag="$1"
  local model="$2"
  local wrapper="$3"
  local output="$SUMMARY_ROOT/$tag"
  if [[ -e "$output/ALL_COMPLETE" ]]; then
    echo "[skip] $tag is complete"
    return
  fi
  MODEL="$model" \
  MODEL_TAG="$tag" \
  PROMPT_WRAPPER="$wrapper" \
  RUN_ROOT="$output" \
  QKSIEVE_GPUS="$GPUS" \
    bash "$LAUNCHER" >"$SUMMARY_ROOT/logs/$tag.log" 2>&1
}

ensure_model "$LLAMA_MODEL"
ensure_model "$QWEN_MODEL" "Qwen/Qwen3-4B-Instruct-2507"
ensure_model "$THIRD_MODEL" "$THIRD_REPO"

run_model llama31_8b "$LLAMA_MODEL" llama3
run_model qwen3_4b "$QWEN_MODEL" qwen3
run_model mistral_7b "$THIRD_MODEL" tokenizer_chat

"$PYTHON" - "$SUMMARY_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tags = ("llama31_8b", "qwen3_4b", "mistral_7b")
summaries = {}
for tag in tags:
    path = root / tag / "paired_summary.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    summaries[tag] = json.loads(path.read_text(encoding="utf-8"))

frozen = [summary["frozen_method"] for summary in summaries.values()]
if any(item != frozen[0] for item in frozen[1:]):
    raise SystemExit("frozen method differs across models")
if any(summary["strict_pairs"] != 3750 for summary in summaries.values()):
    raise SystemExit("one or more models lack 3,750 strict LongBench pairs")
if any(summary["tasks"] != 16 for summary in summaries.values()):
    raise SystemExit("one or more models lack all 16 LongBench tasks")
source_hashes = [summary["source_sha256"] for summary in summaries.values()]
if any(item != source_hashes[0] for item in source_hashes[1:]):
    raise SystemExit("runtime source hashes differ across models")

output = {
    "models": {
        tag: {
            "full_macro": summary["full_macro"],
            "qksieve_macro": summary["qksieve_macro"],
            "quality_retention": summary["quality_retention"],
            "quality_retention_95ci": summary["quality_retention_95ci"],
            "paired_online_speedup": summary["paired_online_speedup"],
            "model_path": summary["model_path"],
            "model_identity_sha256": summary["model_identity_sha256"],
            "prompt_wrapper": summary["prompt_wrapper"],
            "strict_pairs": summary["strict_pairs"],
            "tasks": summary["tasks"],
        }
        for tag, summary in summaries.items()
    },
    "identical_frozen_method": True,
    "frozen_method": frozen[0],
    "source_sha256": source_hashes[0],
    "minimum_quality_retention": min(
        summary["quality_retention"] for summary in summaries.values()
    ),
}
(root / "multimodel_summary.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(json.dumps(output, indent=2, ensure_ascii=False))
PY

touch "$SUMMARY_ROOT/ALL_COMPLETE"
