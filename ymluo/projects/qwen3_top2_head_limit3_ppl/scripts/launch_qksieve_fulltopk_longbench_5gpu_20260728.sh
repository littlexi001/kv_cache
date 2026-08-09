#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
MODEL_TAG="${MODEL_TAG:-llama31_8b}"
PROMPT_WRAPPER="${PROMPT_WRAPPER:-llama3}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_fulltopk_longbench_${MODEL_TAG}_official_middle_paired_5gpu}"
LOG_ROOT=$RUN_ROOT/logs
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHOD=qksieve_fullprompt_auto_plain_fulltopk
METHODS=full_kv,$METHOD

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  echo "QKSIEVE_GPUS must contain exactly five comma-separated GPU ids" >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpus[@]}"; do
  if [[ ! "$gpu" =~ ^[0-5]$ ]]; then
    echo "QKSIEVE_GPUS is restricted to physical GPUs 0-5; got $gpu" >&2
    exit 2
  fi
  if [[ -n "${seen_gpus[$gpu]+x}" ]]; then
    echo "QKSIEVE_GPUS contains duplicate GPU id $gpu" >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done

CUDA_VISIBLE_DEVICES=${gpus[0]} "$PYTHON" -u \
  src/run_sample_calibrated_longbench_20260717.py \
  --model_name_or_path "$MODEL" \
  --longbench_data_dir "$DATA" \
  --output_dir "$RUN_ROOT/smoke" \
  --tasks narrativeqa \
  --methods "$METHODS" \
  --max_samples_per_task 1 \
  --num_shards 1 \
  --shard_index 0 \
  --max_prompt_tokens 7500 \
  --prompt_truncation_mode official_middle \
  --official_query_tail_tokens 8 \
  --max_context_tokens 0 \
  --max_new_tokens_override 8 \
  --prefill_chunk_tokens 2048 \
  --prompt_wrapper "$PROMPT_WRAPPER" \
  --qk_metric_query_shrinkage 0.75 \
  --dtype float16 \
  --device cuda \
  --device_map auto \
  >"$LOG_ROOT/smoke.log" 2>&1

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 2, len(rows)
assert {row["method"] for row in rows} == {
    "full_kv",
    "qksieve_fullprompt_auto_plain_fulltopk",
}
assert len({(row["task"], row["sample_id"]) for row in rows}) == 1
assert {int(row["suffix_tokens"]) for row in rows} == {8}
assert max(int(row["prompt_tokens"]) for row in rows) <= 7512
print("QKSieve true-fulltopk LongBench smoke passed")
PY

pids=()
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$PYTHON" -u \
    src/run_sample_calibrated_longbench_20260717.py \
    --model_name_or_path "$MODEL" \
    --longbench_data_dir "$DATA" \
    --output_dir "$RUN_ROOT/shard${shard}" \
    --tasks "$TASKS" \
    --methods "$METHODS" \
    --max_samples_per_task 0 \
    --num_shards 5 \
    --shard_index "$shard" \
    --max_prompt_tokens 7500 \
    --prompt_truncation_mode official_middle \
    --official_query_tail_tokens 8 \
    --max_context_tokens 0 \
    --max_new_tokens_override 0 \
    --prefill_chunk_tokens 2048 \
    --prompt_wrapper "$PROMPT_WRAPPER" \
    --qk_metric_query_shrinkage 0.75 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    >"$LOG_ROOT/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more QKSieve LongBench shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" "$MODEL" "$MODEL_TAG" "$PROMPT_WRAPPER" "$ROOT" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

run_root = Path(sys.argv[1])
model_path = sys.argv[2]
model_tag = sys.argv[3]
prompt_wrapper = sys.argv[4]
project_root = Path(sys.argv[5]).resolve()

def load(glob):
    rows = []
    for path in sorted(glob):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows

rows = load(run_root.glob("shard[0-9]*/sample_results.csv"))
expected_methods = {
    "full_kv",
    "qksieve_fullprompt_auto_plain_fulltopk",
}
counts = Counter(row["method"] for row in rows)
assert len(rows) == 7500, (len(rows), counts)
assert counts == Counter({method: 3750 for method in expected_methods}), counts
assert len({row["task"] for row in rows}) == 16
assert max(int(row["prompt_tokens"]) for row in rows) <= 7512
assert {int(row["suffix_tokens"]) for row in rows} == {8}
for row in rows:
    if row["method"] != "qksieve_fullprompt_auto_plain_fulltopk":
        continue
    assert row["executed_path"] == row["method"]
    assert row["configured_score_mode"] == (
        "pca_hierarchical_autoqmsetotal15z_"
        "qkmetric_packed_fulltopk"
    )
    assert float(row["sampled_quantile_fallback"]) == 0.0
    assert abs(float(row["configured_index_bits_per_token"]) - 240.0) < 1.0e-6
    assert row["prediction"] is not None

def key(row):
    return row["task"], row["sample_id"]

ours_by_key = {
    key(row): row
    for row in rows
    if row["method"] == "qksieve_fullprompt_auto_plain_fulltopk"
}
full_by_key = {
    key(row): row for row in rows if row["method"] == "full_kv"
}
assert ours_by_key.keys() == full_by_key.keys()

task_rows = defaultdict(list)
for sample_key in sorted(ours_by_key):
    sparse = ours_by_key[sample_key]
    full = full_by_key[sample_key]
    full_generated = max(1, int(full["generated_tokens"]))
    sparse_generated = max(1, int(sparse["generated_tokens"]))
    task_rows[sparse["task"]].append(
        {
            "full_score": float(full["score"]),
            "sparse_score": float(sparse["score"]),
            "full_query": float(full["query_seconds"]),
            "sparse_query": float(sparse["query_seconds"]),
            "full_decode": float(full["decode_seconds"]),
            "sparse_decode": float(sparse["decode_seconds"]),
            "full_online": float(full["online_seconds"]),
            "sparse_online": float(sparse["online_seconds"]),
            "full_decode_tpot": (
                float(full["decode_seconds"]) / full_generated
            ),
            "sparse_decode_tpot": (
                float(sparse["decode_seconds"]) / sparse_generated
            ),
            "full_online_tpot": (
                float(full["online_seconds"]) / full_generated
            ),
            "sparse_online_tpot": (
                float(sparse["online_seconds"]) / sparse_generated
            ),
        }
    )

per_task = {}
for task, paired in sorted(task_rows.items()):
    def mean(field):
        return sum(row[field] for row in paired) / len(paired)

    full_score = mean("full_score")
    sparse_score = mean("sparse_score")
    per_task[task] = {
        "samples": len(paired),
        "full": full_score,
        "qksieve": sparse_score,
        "relative_full": sparse_score / full_score if full_score else None,
        "full_query_seconds": mean("full_query"),
        "qksieve_query_seconds": mean("sparse_query"),
        "full_decode_seconds": mean("full_decode"),
        "qksieve_decode_seconds": mean("sparse_decode"),
        "full_online_seconds": mean("full_online"),
        "qksieve_online_seconds": mean("sparse_online"),
        "full_decode_tpot_seconds": mean("full_decode_tpot"),
        "qksieve_decode_tpot_seconds": mean("sparse_decode_tpot"),
        "full_online_tpot_seconds": mean("full_online_tpot"),
        "qksieve_online_tpot_seconds": mean("sparse_online_tpot"),
        "paired_online_speedup": (
            mean("full_online_tpot") / mean("sparse_online_tpot")
            if mean("sparse_online_tpot") > 0
            else None
        ),
    }

full_macro = sum(item["full"] for item in per_task.values()) / len(per_task)
sparse_macro = sum(item["qksieve"] for item in per_task.values()) / len(per_task)

try:
    import numpy as np

    rng = np.random.default_rng(20260728)
    bootstrap_differences = []
    bootstrap_retentions = []
    task_arrays = {
        task: (
            np.asarray([row["full_score"] for row in paired], dtype=np.float64),
            np.asarray([row["sparse_score"] for row in paired], dtype=np.float64),
        )
        for task, paired in task_rows.items()
    }
    for _ in range(10_000):
        full_means = []
        sparse_means = []
        for full_values, sparse_values in task_arrays.values():
            indices = rng.integers(0, len(full_values), size=len(full_values))
            full_means.append(float(full_values[indices].mean()))
            sparse_means.append(float(sparse_values[indices].mean()))
        boot_full = sum(full_means) / len(full_means)
        boot_sparse = sum(sparse_means) / len(sparse_means)
        bootstrap_differences.append(boot_sparse - boot_full)
        bootstrap_retentions.append(boot_sparse / boot_full)
    difference_ci = [
        float(np.quantile(bootstrap_differences, 0.025)),
        float(np.quantile(bootstrap_differences, 0.975)),
    ]
    retention_ci = [
        float(np.quantile(bootstrap_retentions, 0.025)),
        float(np.quantile(bootstrap_retentions, 0.975)),
    ]
except ImportError:
    difference_ci = None
    retention_ci = None

def paired_mean(field):
    values = [
        row[field]
        for paired in task_rows.values()
        for row in paired
    ]
    return sum(values) / len(values)

source_paths = [
    project_root / "src/run_sample_calibrated_longbench_20260717.py",
    project_root / "src/run_controlled_public_kv_benchmark_v1.py",
    project_root / "src/run_head_top2_targeted_ppl_20260714.py",
    project_root / "src/variablebit_spectral_cuda_20260727.py",
    project_root / "src/qabs_cuda_kernels.py",
]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model_root = Path(model_path)
model_identity_paths = [
    path
    for path in (
        model_root / "config.json",
        model_root / "tokenizer_config.json",
        model_root / "generation_config.json",
        model_root / "model.safetensors.index.json",
    )
    if path.is_file()
]

summary = {
    "strict_pairs": len(ours_by_key),
    "tasks": len(per_task),
    "model_path": model_path,
    "model_tag": model_tag,
    "model_identity_sha256": {
        path.name: sha256(path) for path in model_identity_paths
    },
    "prompt_wrapper": prompt_wrapper,
    "full_macro": full_macro,
    "qksieve_macro": sparse_macro,
    "quality_retention": sparse_macro / full_macro,
    "macro_difference_95ci": difference_ci,
    "quality_retention_95ci": retention_ci,
    "mean_full_query_seconds": paired_mean("full_query"),
    "mean_qksieve_query_seconds": paired_mean("sparse_query"),
    "mean_full_decode_seconds": paired_mean("full_decode"),
    "mean_qksieve_decode_seconds": paired_mean("sparse_decode"),
    "mean_full_online_seconds": paired_mean("full_online"),
    "mean_qksieve_online_seconds": paired_mean("sparse_online"),
    "mean_full_decode_tpot_seconds": paired_mean("full_decode_tpot"),
    "mean_qksieve_decode_tpot_seconds": paired_mean("sparse_decode_tpot"),
    "mean_full_online_tpot_seconds": paired_mean("full_online_tpot"),
    "mean_qksieve_online_tpot_seconds": paired_mean("sparse_online_tpot"),
    "paired_online_speedup": (
        paired_mean("full_online_tpot") / paired_mean("sparse_online_tpot")
        if paired_mean("sparse_online_tpot") > 0
        else None
    ),
    "timing_protocol": {
        "paired_online_speedup_metric": (
            "(dense suffix + decode) seconds per actually generated token"
        ),
        "fixed_horizon_system_speed_reported_separately": True,
    },
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
    "per_task": per_task,
}
(run_root / "paired_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

touch "$RUN_ROOT/ALL_COMPLETE"
