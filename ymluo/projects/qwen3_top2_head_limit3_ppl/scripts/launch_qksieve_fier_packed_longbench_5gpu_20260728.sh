#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/models/LLM-Research-Meta-Llama-3.1-8B-Instruct-ms}"
MODEL_TAG="${MODEL_TAG:-llama31_8b}"
PROMPT_WRAPPER="${PROMPT_WRAPPER:-llama3}"
DATA="${DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results/20260728_qksieve_fier_packed_longbench_${MODEL_TAG}_paired_5gpu}"
LOG_ROOT="$RUN_ROOT/logs"
TASKS=narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,trec,triviaqa,samsum,passage_retrieval_en,passage_count,gov_report,multi_news,lcc,repobench-p
METHODS="${METHODS:-full_kv,qksieve_fullprompt_auto_plain_fulltopk,fier_rtn1_g32_packed_fulltopk}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-0}"
EXPECTED_PAIRS="${EXPECTED_PAIRS:-3750}"

export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$ROOT/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

IFS=',' read -r -a gpus <<< "${QKSIEVE_GPUS:-0,1,2,3,4}"
if [[ "${#gpus[@]}" -ne 5 ]]; then
  echo "QKSIEVE_GPUS must contain exactly five GPU ids" >&2
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
  src/validate_fier_rtn1_cuda_20260728.py \
  >"$LOG_ROOT/fier_cuda_validation.log" 2>&1

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

"$PYTHON" - "$RUN_ROOT/smoke/sample_results.csv" "$METHODS" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
expected = {item for item in sys.argv[2].split(",") if item}
assert len(rows) == len(expected), len(rows)
assert {row["method"] for row in rows} == expected
assert len({(row["task"], row["sample_id"]) for row in rows}) == 1
assert {int(row["suffix_tokens"]) for row in rows} == {8}
assert max(int(row["prompt_tokens"]) for row in rows) <= 7512
print("packed FIER/QKSieve LongBench smoke passed")
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
    --max_samples_per_task "$MAX_SAMPLES_PER_TASK" \
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
  echo "one or more fair-comparison shards failed; valid rows remain" >&2
  exit 1
fi

"$PYTHON" src/summarize_paired_longbench_20260728.py \
  --run_root "$RUN_ROOT" \
  --methods "$METHODS" \
  --reference_method full_kv \
  --expected_pairs "$EXPECTED_PAIRS" \
  --expected_tasks 16 \
  --bootstrap_resamples 10000 \
  --output "$RUN_ROOT/paired_summary.json" \
  >"$LOG_ROOT/summary.log" 2>&1

"$PYTHON" - "$ROOT" "$RUN_ROOT" "$MODEL" "$MODEL_TAG" "$PROMPT_WRAPPER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
source_paths = [
    root / "src/run_sample_calibrated_longbench_20260717.py",
    root / "src/run_head_top2_targeted_ppl_20260714.py",
    root / "src/variablebit_spectral_cuda_20260727.py",
    root / "src/fier_rtn1_cuda_20260728.py",
    root / "src/qabs_cuda_kernels.py",
]
manifest = {
    "model_path": sys.argv[3],
    "model_tag": sys.argv[4],
    "prompt_wrapper": sys.argv[5],
    "protocol": {
        "prompt_truncation": "official_middle",
        "prompt_limit": 7500,
        "query_tail_tokens": 8,
        "fallback": False,
        "rerank": False,
        "recent_or_sink_reservation": False,
    },
    "shared_budget": "min(N, 1280, max(256, ceil(0.06*N)))",
    "qksieve": {
        "score_mode": (
            "pca_hierarchical_autoqmsetotal15z_"
            "qkmetric_packed_fulltopk"
        ),
        "index_bits_per_token_per_kv_head": 240,
        "query_shrinkage": 0.75,
    },
    "fier": {
        "score_mode": "fier_rtn1_g32_packed_fulltopk",
        "sequence_group_size": 32,
        "index_bits_per_token_per_kv_head": 256,
        "implementation": "audited paper-spec port; not official code",
    },
    "controlled_uniform1_ablation": {
        "key_pca_score_mode": (
            "pca_hierarchical_fixed11111111_packed_fulltopk"
        ),
        "qk_balanced_score_mode": (
            "pca_hierarchical_fixed11111111_"
            "qkmetric_packed_fulltopk"
        ),
        "index_bits_per_token_per_kv_head": 256,
    },
    "causal_ablation_modes": {
        "random_rotation_uniform1": (
            "pca_hierarchical_fixed11111111_"
            "random_packed_fulltopk"
        ),
        "without_query_covariance": (
            "pca_hierarchical_autokeytotal15z_packed_fulltopk"
        ),
        "without_query_covariance_in_allocation": (
            "pca_hierarchical_autokeytotal15z_"
            "qkmetric_packed_fulltopk"
        ),
        "shared_prompt_path": (
            "dense question/instruction suffix; sparse answer decode"
        ),
    },
    "shared_final_attention": "qabs_cuda_kernels exact sparse attention",
    "source_sha256": {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in source_paths
    },
}
(run_root / "method_contract.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
PY

touch "$RUN_ROOT/ALL_COMPLETE"
