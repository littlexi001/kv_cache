#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary}"
PY="${PY:-/home/fdong/miniconda3/envs/moe/bin/python}"
MODEL="${MODEL:-/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
ROOT="${ROOT:-$PROJECT/outputs/attention_confidence_qwen3_8b_english_mechanism_probes_20260718}"
BASELINE="${BASELINE:-$PROJECT/outputs/attention_confidence_qwen3_8b_english_single_token_128k_20260718}"
LENGTHS="${LENGTHS:-8000,32000,64000,96000,128000}"

while [[ ! -f "$ROOT/probes.done" ]]; do
  if [[ -f "$ROOT/probes.failed" ]]; then
    echo "main mechanism probes failed; cloze probes not started" >&2
    exit 1
  fi
  sleep 60
done

rm -f "$ROOT/cloze.done" "$ROOT/cloze.failed"
run_cloze() {
  local label="$1"
  local devices="$2"
  local query_mode="$3"
  local output="$ROOT/$label"
  mkdir -p "$output/data"
  CUDA_VISIBLE_DEVICES="$devices" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" -u "$PROJECT/src/run_attention_confidence_sweep_8b.py" \
    --model_name_or_path "$MODEL" \
    --output_dir "$output" \
    --lengths "$LENGTHS" \
    --seed 0 \
    --code_mode english_single_token \
    --placement middle \
    --query_mode "$query_mode" \
    --prompt_style cloze \
    --max_top 100 \
    --prefill_chunk_size 128 \
    --dtype float16 \
    --device_map balanced \
    --attn_implementation sdpa \
    --original_max_position_embeddings 40960 \
    --global_max_position 130000 \
    --shard_label "$label" \
    >"$ROOT/logs/$label.log" 2>&1
  "$PY" "$PROJECT/src/analyze_attention_confidence_sweep.py" --output_dir "$output"
}

run_cloze middle_hop1_cloze 0,1 hop1 & first=$!
run_cloze middle_oracle_hop2_cloze 2,3 oracle_hop2 & second=$!
failed=0
wait "$first" || failed=1
wait "$second" || failed=1
if [[ "$failed" -ne 0 ]]; then
  date -Is > "$ROOT/cloze.failed"
  exit 1
fi

"$PY" "$PROJECT/src/analyze_english_length_mechanisms.py" \
  --baseline_dir "$BASELINE" \
  --probes_root "$ROOT" \
  --output_dir "$ROOT/combined_analysis"
date -Is > "$ROOT/cloze.done"
