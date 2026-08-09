#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
ROOT="${ROOT:-$PROJECT/outputs/attention_confidence_qwen3_8b_english_mechanism_probes_20260718}"
BASELINE="${BASELINE:-$PROJECT/outputs/attention_confidence_qwen3_8b_english_single_token_128k_20260718}"
LENGTHS="${LENGTHS:-8000,32000,64000,96000,128000}"

mkdir -p "$ROOT/logs"
rm -f "$ROOT/probes.done" "$ROOT/probes.failed"

run_probe() {
  local label="$1"
  local devices="$2"
  local placement="$3"
  local query_mode="$4"
  local global_max_position="${5:-130000}"
  local run_lengths="${6:-$LENGTHS}"
  local output="$ROOT/$label"
  mkdir -p "$output/data"
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --lengths "$run_lengths" \
    --seed 0 \
    --code_mode english_single_token \
    --placement "$placement" \
    --query_mode "$query_mode" \
    --max_top 100 \
    --prefill_chunk_size "${PREFILL_CHUNK_SIZE:-128}" \
    --dtype float16 \
    --device_map balanced \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position "$global_max_position" \
    --shard_label "$label" \
    >"$ROOT/logs/$label.log" 2>&1
  "$PY" "$PROJECT/src/analyze_attention_confidence_sweep.py" --output_dir "$output"
}

declare -a pids labels
labels=(prefix_full2 recent_full2 middle_hop1 middle_oracle_hop2)
run_probe prefix_full2 0,1 prefix full2 & pids[0]=$!
run_probe recent_full2 2,3 recent full2 & pids[1]=$!
run_probe middle_hop1 4,5 middle hop1 & pids[2]=$!
run_probe middle_oracle_hop2 6,7 middle oracle_hop2 & pids[3]=$!

failed=0
for index in 0 1 2 3; do
  if ! wait "${pids[$index]}"; then
    echo "probe ${labels[$index]} failed; inspect $ROOT/logs/${labels[$index]}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  date -Is > "$ROOT/probes.failed"
  exit 1
fi

# RoPE controls distinguish generic length effects from the factor-4 YaRN
# configuration required to reach 128K.  These are small, sequential follow-ups
# after the four main intervention jobs release the GPUs.
declare -a rope_pids rope_labels
rope_labels=(native40k_middle_full2 yarn2_middle_full2)
run_probe native40k_middle_full2 0,1 middle full2 40960 "8000,32000" & rope_pids[0]=$!
run_probe yarn2_middle_full2 2,3 middle full2 65536 "32000,64000" & rope_pids[1]=$!
for index in 0 1; do
  if ! wait "${rope_pids[$index]}"; then
    echo "probe ${rope_labels[$index]} failed; inspect $ROOT/logs/${rope_labels[$index]}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  date -Is > "$ROOT/probes.failed"
  exit 1
fi

"$PY" "$PROJECT/src/analyze_english_length_mechanisms.py" \
  --baseline_dir "$BASELINE" \
  --probes_root "$ROOT" \
  --output_dir "$ROOT/combined_analysis"
date -Is > "$ROOT/probes.done"
