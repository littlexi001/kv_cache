#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
OUT="${OUT:-$PROJECT/outputs/common_tail_boundary_qwen3_8b_fixed_recent_20260722}"
SHORT_LENGTHS="${SHORT_LENGTHS:-1024,4096,8192,16384,32768,49152,65536,81920,98304}"
LONG_LENGTHS="${LONG_LENGTHS:-114688,127500}"

mkdir -p "$OUT/logs"

# The original launcher owns GPU pairs 0-1 and 2-3.  Two unrelated existing
# jobs occupied 4-7 after preflight, so wait until the original launcher has
# collected its two healthy workers and marked the partial run failed.
while [[ ! -f "$OUT/launcher.failed" && ! -f "$OUT/launcher.done" ]]; do
  sleep 30
done
[[ -f "$OUT/launcher.done" ]] && exit 0

run_shard() {
  local stage="$1"
  local shard="$2"
  local shard_count="$3"
  local devices="$4"
  local lengths="$5"
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" -u "$PROJECT/src/run_common_tail_boundary_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUT" \
    --lengths "$lengths" \
    --samples_per_bin 8 \
    --placement fixed_recent \
    --recent_gap 256 \
    --seed 20260721 \
    --shard_index "$shard" \
    --num_shards "$shard_count" \
    --shard_label "${stage}_shard$shard" \
    --prefill_chunk_size 128 \
    --dtype float16 \
    --device_map balanced \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 130000 \
    >"$OUT/logs/${stage}_shard${shard}_resume.log" 2>&1
}

# Resume the two collided short shards without touching GPUs 4-7.
declare -a pids
run_shard short 2 4 0,1 "$SHORT_LENGTHS" & pids[0]=$!
run_shard short 3 4 2,3 "$SHORT_LENGTHS" & pids[1]=$!
status=0
for index in 0 1; do
  if ! wait "${pids[$index]}"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  date -Is >"$OUT/resume.failed"
  exit 1
fi

# 112K/128K require four GPUs to avoid the SDPA repeat_kv transient peak.
# Always reuse 0-3 serially.  This is slower than opportunistically taking all
# eight GPUs, but it cannot race with the unrelated jobs currently using 4-7.
status=0
run_shard long 0 2 0,1,2,3 "$LONG_LENGTHS" || status=1
run_shard long 1 2 0,1,2,3 "$LONG_LENGTHS" || status=1
if [[ "$status" -ne 0 ]]; then
  date -Is >"$OUT/resume.failed"
  exit 1
fi

"$PY" "$PROJECT/src/summarize_common_tail_boundary.py" --output_dir "$OUT" \
  >"$OUT/logs/summarize.log" 2>&1
rm -f "$OUT/launcher.failed" "$OUT/resume.failed"
date -Is >"$OUT/launcher.done"
echo "completed: $OUT/report.md"
