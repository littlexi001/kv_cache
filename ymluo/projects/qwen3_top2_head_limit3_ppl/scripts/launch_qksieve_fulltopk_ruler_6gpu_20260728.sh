#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms
LM_EVAL=/home/fdong/lm-evaluation-harness
SHORT_DATA=$ROOT/data/ruler_generated/llama31_8b_ruler13_4k32k_m10_seed42.jsonl
LONG_DATA=$ROOT/data/ruler_generated/llama31_8b_ruler13_64k128k_m5_seed42.jsonl
RUN_ROOT=$ROOT/results/20260728_qksieve_fulltopk_ruler_6gpu
LOG_ROOT=$RUN_ROOT/logs
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
METHODS=full_kv,qksieve_fullprompt_auto_plain_fulltopk

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

prepare_ruler_data() {
  local output="$1"
  local lengths="$2"
  local samples="$3"
  "$PYTHON" src/prepare_hierarchical_ruler_data_20260716.py \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --output "$output" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths "$lengths" \
    --max_samples_per_task "$samples" \
    --seed 42
}

prepare_ruler_data "$SHORT_DATA" 4096,8192,16384,32768 10 \
  >"$LOG_ROOT/prepare_short.log" 2>&1
prepare_ruler_data "$LONG_DATA" 65536,131072 5 \
  >"$LOG_ROOT/prepare_long.log" 2>&1

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
  src/run_sample_calibrated_ruler_20260717.py \
  --model_name_or_path "$MODEL" \
  --examples_jsonl "$SHORT_DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --methods "$METHODS" \
  --ruler_tasks niah_single_1 \
  --ruler_lengths 4096 \
  --max_samples_per_task 1 \
  --max_new_tokens_override 16 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper llama3 \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

run_short_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$SHORT_DATA" \
    --output_dir "$RUN_ROOT/short_shard${shard}" \
    --methods "$METHODS" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 4096,8192,16384,32768 \
    --max_samples_per_task 10 \
    --num_shards 6 \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/short_shard${shard}.log" 2>&1
}

short_pids=()
for shard in 0 1 2 3 4 5; do
  run_short_shard "$shard" "$shard" &
  short_pids+=("$!")
done
failed=0
for pid in "${short_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more short RULER shards failed; valid rows remain" >&2
  exit 1
fi

# Llama-3.1-8B at 128K needs four 24GB cards in this environment.
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -u \
    src/run_sample_calibrated_ruler_20260717.py \
    --model_name_or_path "$MODEL" \
    --examples_jsonl "$LONG_DATA" \
    --output_dir "$RUN_ROOT/long_shard${shard}" \
    --methods "$METHODS" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 65536,131072 \
    --max_samples_per_task 5 \
    --num_shards 2 \
    --shard_index "$shard" \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper llama3 \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map balanced \
    >"$LOG_ROOT/long_shard${shard}.log" 2>&1
done

"$PYTHON" src/summarize_countcap_benchmark_20260722.py \
  --kind ruler \
  --input_glob "$RUN_ROOT/*_shard[0-9]*/sample_results.csv" \
  --output_dir "$RUN_ROOT/merged" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$RUN_ROOT/merged/sample_results.csv" "$RUN_ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
project_root = run_root.parents[1]
with csv_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

expected = {"full_kv", "qksieve_fullprompt_auto_plain_fulltopk"}
counts = Counter(row["method"] for row in rows)
pairs = defaultdict(set)
for row in rows:
    pairs[(row["task"], row["sample_id"])].add(row["method"])

assert len(rows) == 1300, (len(rows), counts)
assert counts == Counter({method: 650 for method in expected}), counts
assert len(pairs) == 650
assert all(methods == expected for methods in pairs.values())
assert len({row["base_task"] for row in rows}) == 13
assert {int(row["requested_length"]) for row in rows} == {
    4096, 8192, 16384, 32768, 65536, 131072
}

grouped = defaultdict(dict)
for row in rows:
    grouped[
        (
            row["base_task"],
            int(row["requested_length"]),
            row["sample_id"],
        )
    ][row["method"]] = row
assert all(set(methods) == expected for methods in grouped.values())

cells = defaultdict(list)
for (task, length, _), methods in grouped.items():
    full = methods["full_kv"]
    sparse = methods["qksieve_fullprompt_auto_plain_fulltopk"]
    cells[(task, length)].append(
        {
            "full_score": float(full["score"]),
            "sparse_score": float(sparse["score"]),
            "full_online": float(full["online_seconds"]),
            "sparse_online": float(sparse["online_seconds"]),
            "full_decode": float(full["decode_seconds"]),
            "sparse_decode": float(sparse["decode_seconds"]),
        }
    )

per_task_length = {}
for (task, length), paired in sorted(cells.items()):
    def mean(field):
        return sum(row[field] for row in paired) / len(paired)

    full_score = mean("full_score")
    sparse_score = mean("sparse_score")
    per_task_length[f"{task}@{length}"] = {
        "task": task,
        "length": length,
        "samples": len(paired),
        "full": full_score,
        "qksieve": sparse_score,
        "relative_full": sparse_score / full_score if full_score else None,
        "full_online_seconds": mean("full_online"),
        "qksieve_online_seconds": mean("sparse_online"),
        "paired_online_speedup": (
            mean("full_online") / mean("sparse_online")
            if mean("sparse_online") > 0
            else None
        ),
        "paired_decode_speedup": (
            mean("full_decode") / mean("sparse_decode")
            if mean("sparse_decode") > 0
            else None
        ),
    }

per_length = {}
for length in sorted({key[1] for key in cells}):
    entries = [
        value
        for value in per_task_length.values()
        if value["length"] == length
    ]
    full_macro = sum(item["full"] for item in entries) / len(entries)
    sparse_macro = sum(item["qksieve"] for item in entries) / len(entries)
    per_length[str(length)] = {
        "tasks": len(entries),
        "full_macro": full_macro,
        "qksieve_macro": sparse_macro,
        "quality_retention": (
            sparse_macro / full_macro if full_macro else None
        ),
        "geomean_online_speedup": (
            __import__("math").exp(
                sum(
                    __import__("math").log(item["paired_online_speedup"])
                    for item in entries
                    if item["paired_online_speedup"] is not None
                )
                / len(entries)
            )
        ),
    }

source_paths = [
    project_root / "src/run_sample_calibrated_ruler_20260717.py",
    project_root / "src/run_sample_calibrated_longbench_20260717.py",
    project_root / "src/run_controlled_public_kv_benchmark_v1.py",
    project_root / "src/run_head_top2_targeted_ppl_20260714.py",
    project_root / "src/variablebit_spectral_cuda_20260727.py",
    project_root / "src/qabs_cuda_kernels.py",
    project_root / "src/qksieve_query_cuda_20260728.py",
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "strict_pairs": len(grouped),
    "rows": len(rows),
    "tasks": len({key[0] for key in grouped}),
    "lengths": sorted({key[1] for key in grouped}),
    "frozen_method": {
        "method": "qksieve_fullprompt_auto_plain_fulltopk",
        "score_mode": (
            "pca_hierarchical_autoqmsetotal15z_"
            "qkmetric_packed_fulltopk"
        ),
        "query_shrinkage": 0.75,
        "query_tail_tokens": 8,
        "index_bits_per_token_per_kv_head": 240,
        "budget": "min(N, 1280, max(256, ceil(0.06*N)))",
        "proxy_topk_dtype": "float32",
        "exact_kv_dtype": "float16",
        "rerank": False,
        "fallback": False,
        "recent_or_sink_reservation": False,
    },
    "source_sha256": {
        str(path.relative_to(project_root)): sha256(path)
        for path in source_paths
    },
    "per_length": per_length,
    "per_task_length": per_task_length,
}
(run_root / "paired_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print("validated 650 strict two-way QKSieve RULER samples")
PY

"$PYTHON" src/summarize_qksieve_ruler_20260728.py \
  --input_csv "$RUN_ROOT/merged/sample_results.csv" \
  --project_root "$ROOT" \
  --output "$RUN_ROOT/paired_summary.json" \
  --bootstrap_resamples 10000 \
  --seed 20260728 \
  >"$LOG_ROOT/paired_summary.log" 2>&1

touch "$RUN_ROOT/ALL_COMPLETE"
