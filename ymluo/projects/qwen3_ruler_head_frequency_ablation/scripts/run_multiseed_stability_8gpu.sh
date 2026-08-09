#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_ruler_head_frequency_ablation
TOP2=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
PY=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LM_EVAL=/home/fdong/lm-evaluation-harness
HOTPot=/home/fdong/ymluo/datasets/ruler_sources/hotpotqa/distractor/validation-00000-of-00001.parquet
TASKS=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_squad,qa_hotpot
RUN_NAME=${1:-multiseed_frequency_scaling_20260806}
RUN=$ROOT/outputs/$RUN_NAME
mkdir -p "$RUN/data" "$RUN/specs"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

prepare_seed() {
  local seed=$1
  local samples=$2
  local output="$RUN/data/ruler32k_seed${seed}_m${samples}.jsonl"
  "$PY" "$TOP2/src/prepare_hierarchical_ruler_data_20260716.py" \
    --model_name_or_path "$MODEL" \
    --lm_eval_path "$LM_EVAL" \
    --output "$output" \
    --ruler_tasks "$TASKS" \
    --ruler_lengths 32768 \
    --max_samples_per_task "$samples" \
    --seed "$seed" \
    --ruler_hotpot_parquet "$HOTPot"
}

run_seed() {
  local phase=$1
  local seed=$2
  local samples=$3
  local specs=$4
  local outroot="$RUN/$phase/seed$seed"
  if test -f "$outroot/stage.done"; then return; fi
  mkdir -p "$outroot"
  local data="$RUN/data/ruler32k_seed${seed}_m${samples}.jsonl"
  local spec_count
  spec_count=$($PY -c "import json; print(len(json.load(open('$specs'))['specs']))")
  local shards=8
  if test "$spec_count" -lt "$shards"; then shards=$spec_count; fi
  local pids=()
  for shard in $(seq 0 $((shards - 1))); do
    local out="$outroot/shard$shard"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$shard" TOKENIZERS_PARALLELISM=false \
      "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
      --model-name-or-path "$MODEL" \
      --examples-jsonl "$data" \
      --specs-json "$specs" \
      --output-dir "$out" \
      --target-length 32768 \
      --max-new-tokens-cap 128 \
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
  "$PY" "$ROOT/src/summarize_sweep.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
  touch "$outroot/stage.done"
}

# Generate all data before using GPUs.
prepare_seed 43 1
prepare_seed 44 1
prepare_seed 45 2
prepare_seed 46 2
prepare_seed 47 2

"$PY" "$ROOT/src/make_stability_specs.py" --mode smoke --output "$RUN/specs/smoke.json"
"$PY" "$ROOT/src/make_stability_specs.py" --mode validation --output "$RUN/specs/validation.json"

# Validate the rewritten forward and both alpha endpoints on one unseen sample.
SMOKE="$RUN/smoke"
if ! test -f "$RUN/smoke.done"; then
  mkdir -p "$SMOKE/shard0"
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
    "$PY" -u "$ROOT/src/run_frequency_sweep.py" \
    --model-name-or-path "$MODEL" \
    --examples-jsonl "$RUN/data/ruler32k_seed43_m1.jsonl" \
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
assert len(rows) == 4, len(rows)
replay=[row for row in rows if row['variant']=='patched_native_replay'][0]
assert replay['patched_vs_original_max_logit_error'] == 0.0, replay['patched_vs_original_max_logit_error']
assert all(row['finite_logits'] and math.isfinite(row['gold_answer_mean_nll']) for row in rows)
PY
  touch "$RUN/smoke.done"
fi

# Validation seeds are the only data used to choose alpha/configuration.
run_seed validation 43 1 "$RUN/specs/validation.json"
run_seed validation 44 1 "$RUN/specs/validation.json"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "43=$RUN/validation/seed43" \
  --seed-run "44=$RUN/validation/seed44" \
  --output-dir "$RUN/validation/combined" \
  >"$RUN/validation/combined_stdout.log" 2>"$RUN/validation/combined_stderr.log"
"$PY" "$ROOT/src/select_test_specs.py" \
  --validation-summary "$RUN/validation/combined/summary.json" \
  --output "$RUN/specs/test.json" --limit 3 \
  >"$RUN/specs/test_selection.log"
touch "$RUN/validation.done"

# Test once on three unseen seeds after the spec is frozen.
run_seed test 45 2 "$RUN/specs/test.json"
run_seed test 46 2 "$RUN/specs/test.json"
run_seed test 47 2 "$RUN/specs/test.json"
"$PY" "$ROOT/src/summarize_multiseed.py" \
  --seed-run "45=$RUN/test/seed45" \
  --seed-run "46=$RUN/test/seed46" \
  --seed-run "47=$RUN/test/seed47" \
  --output-dir "$RUN/test/combined" \
  >"$RUN/test/combined_stdout.log" 2>"$RUN/test/combined_stderr.log"
touch "$RUN/test.done"
touch "$RUN/launcher.done"
trap - EXIT
