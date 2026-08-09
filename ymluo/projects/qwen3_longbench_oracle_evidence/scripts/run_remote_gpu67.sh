#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/fdong/ymluo/projects/qwen3_longbench_oracle_evidence}"
PYTHON="${PYTHON:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
LONG_BENCH_DATA="${LONG_BENCH_DATA:-/home/fdong/ymluo/external/KVCache-Factory/data/LongBench}"
RUN_TAG="${RUN_TAG:-hotpot_gold_evidence_pilot_20260802}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/${RUN_TAG}}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-6000}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAMPLE_SEED="${SAMPLE_SEED:-20260802}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-sentence_exact}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-position_length}"
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-100000}"

mkdir -p "${OUTPUT_ROOT}/shard0" "${OUTPUT_ROOT}/shard1" "${OUTPUT_ROOT}/merged"

COMMON=(
  "${ROOT}/src/run_hotpot_oracle_pilot.py"
  --model_name_or_path "${MODEL}"
  --longbench_jsonl "${LONG_BENCH_DATA}/hotpotqa.jsonl"
  --max_samples "${MAX_SAMPLES}"
  --min_prompt_tokens "${MIN_PROMPT_TOKENS}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --random_seeds 0,1,2
  --sample_seed "${SAMPLE_SEED}"
  --alignment_mode "${ALIGNMENT_MODE}"
  --sample_strategy "${SAMPLE_STRATEGY}"
  --shard_count 2
  --dtype bfloat16
  --attn_implementation sdpa
)
if [[ -n "${EXCLUDE_MANIFEST}" ]]; then
  COMMON+=(--exclude_manifest "${EXCLUDE_MANIFEST}")
fi

env CUDA_VISIBLE_DEVICES=6 "${PYTHON}" -u "${COMMON[@]}" \
  --shard_index 0 --output_dir "${OUTPUT_ROOT}/shard0" \
  >"${OUTPUT_ROOT}/shard0/run.log" 2>&1 &
PID0=$!

env CUDA_VISIBLE_DEVICES=7 "${PYTHON}" -u "${COMMON[@]}" \
  --shard_index 1 --output_dir "${OUTPUT_ROOT}/shard1" \
  >"${OUTPUT_ROOT}/shard1/run.log" 2>&1 &
PID1=$!

status=0
wait "${PID0}" || status=1
wait "${PID1}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "At least one inference shard failed. Inspect ${OUTPUT_ROOT}/shard*/run.log" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/src/summarize_hotpot_oracle_pilot.py" \
  --shard_dirs "${OUTPUT_ROOT}/shard0" "${OUTPUT_ROOT}/shard1" \
  --output_dir "${OUTPUT_ROOT}/merged" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
  >"${OUTPUT_ROOT}/merged/summarize.log" 2>&1

echo "Complete: ${OUTPUT_ROOT}/merged"
