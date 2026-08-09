#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
DATA=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
SCREEN_IDS=niah_multikey_3_32768_0,fwe_32768_0,cwe_32768_0,niah_multivalue_32768_0,qa_squad_32768_1,qa_hotpot_32768_0
RUN_NAME=${1:-deep_head_frequency_sweep_20260805}
RUN=$ROOT/outputs/$RUN_NAME
mkdir -p "$RUN/specs"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

run_stage() {
  local stage_name=$1
  local specs_path=$2
  local sample_ids=$3
  local max_new=$4
  local stage_dir="$RUN/$stage_name"
  mkdir -p "$stage_dir"
  local spec_count
  spec_count=$($PY -c "import json; print(len(json.load(open('$specs_path'))['specs']))")
  local shards=8
  if test "$spec_count" -lt "$shards"; then shards=$spec_count; fi
  local pids=()
  for shard in $(seq 0 $((shards - 1))); do
    local out="$stage_dir/shard$shard"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$DATA" \
      --specs-json "$specs_path" \
      --output-dir "$out" \
      --sample-ids "$sample_ids" \
      --target-length 32768 \
      --max-new-tokens-cap "$max_new" \
      --prefill-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --spec-shard-count "$shards" \
      --spec-shard-index "$shard" \
      >"$out/stdout.log" 2>"$out/stderr.log" &
    echo $! >"$out/pid.txt"
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$stage_dir" \
    >"$stage_dir/summary_stdout.log" 2>"$stage_dir/summary_stderr.log"
  touch "$stage_dir/stage.done"
}

# Native forward equivalence and one finite intervention check.
"$PY" "$ROOT/src/make_specs.py" --stage smoke --output "$RUN/specs/smoke.json"
SMOKE="$RUN/smoke"
mkdir -p "$SMOKE"
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
  "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
  --model-name-or-path "$MODEL" \
  --examples-jsonl "$DATA" \
  --specs-json "$RUN/specs/smoke.json" \
  --output-dir "$SMOKE/shard0" \
  --sample-ids niah_multikey_3_32768_0 \
  --target-length 32768 \
  --max-new-tokens-cap 32 \
  --prefill-chunk-size 256 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --load-in-4bit \
  >"$SMOKE/stdout.log" 2>"$SMOKE/stderr.log"
"$PY" - "$SMOKE/shard0/rows.jsonl" <<'PY'
import json, math, sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
assert len(rows) == 3, len(rows)
replay=[row for row in rows if row['variant']=='patched_native_replay'][0]
assert replay['patched_vs_original_max_logit_error'] is not None
assert replay['patched_vs_original_max_logit_error'] < 1e-4, replay['patched_vs_original_max_logit_error']
assert all(row['finite_logits'] and math.isfinite(row['gold_answer_mean_nll']) for row in rows)
PY
touch "$RUN/smoke.done"

# A: 3 deep-layer blocks × 8 GQA groups × 8 coarse frequency bands.
"$PY" "$ROOT/src/make_specs.py" --stage coarse --output "$RUN/specs/coarse.json"
run_stage coarse "$RUN/specs/coarse.json" "$SCREEN_IDS" 64

# B: within the top eight coarse regions, separate layer and frequency axes.
"$PY" "$ROOT/src/make_specs.py" --stage refine \
  --summary "$RUN/coarse/summary.json" --limit 8 --output "$RUN/specs/refine.json"
run_stage refine "$RUN/specs/refine.json" "$SCREEN_IDS" 64

# Cross the best layer and frequency found in each retained region.
"$PY" "$ROOT/src/make_specs.py" --stage cross \
  --summary "$RUN/refine/summary.json" --output "$RUN/specs/cross.json"
run_stage cross "$RUN/specs/cross.json" "$SCREEN_IDS" 64

# C: validate the best eight distinct interventions on all 26 RULER-32K examples.
"$PY" "$ROOT/src/make_specs.py" --stage finalists \
  --summary "$RUN/coarse/summary.json" \
  --summary "$RUN/refine/summary.json" \
  --summary "$RUN/cross/summary.json" \
  --limit 8 --output "$RUN/specs/finalists.json"
run_stage finalists "$RUN/specs/finalists.json" "" 128

# Greedy cumulative combinations, again on the complete 26-example probe.
"$PY" "$ROOT/src/make_specs.py" --stage combinations \
  --summary "$RUN/finalists/summary.json" --limit 8 --output "$RUN/specs/combinations.json"
run_stage combinations "$RUN/specs/combinations.json" "" 128

touch "$RUN/launcher.done"
trap - EXIT
