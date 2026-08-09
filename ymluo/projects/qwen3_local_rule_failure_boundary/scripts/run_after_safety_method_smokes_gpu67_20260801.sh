#!/usr/bin/env bash
set -euo pipefail

# Queue two independent method smokes after the formal safety audit releases
# physical GPUs 6 and 7.  This script never addresses any other GPU.

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SAFETY="$ROOT/outputs/20260801_suppression_certificate_safety_gpu67_v3_gqa"
QUEUE="$ROOT/outputs/20260801_after_safety_method_smokes_gpu67"
VALUE_OUT="$ROOT/outputs/20260801_value_mediated_singleton_smoke_gpu6"
MPR_OUT="$ROOT/outputs/20260801_strict_mpr_token_sparse_smoke_gpu7"

mkdir -p "$QUEUE" "$VALUE_OUT" "$MPR_OUT"

for _ in $(seq 1 720); do
  if [[ -f "$SAFETY/COMPLETE" ]]; then
    break
  fi
  if [[ -f "$SAFETY/FAILED" ]]; then
    printf 'safety_failed\n' > "$QUEUE/FAILED"
    exit 1
  fi
  sleep 10
done

if [[ ! -f "$SAFETY/COMPLETE" ]]; then
  printf 'safety_wait_timeout\n' > "$QUEUE/FAILED"
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=6 \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" "$ROOT/src/run_value_mediated_rope_probe_8b.py" \
  --model-name-or-path "$MODEL" \
  --lengths 8192 \
  --class-sample-count 8 \
  --packet-gap-tokens 16 \
  --anchor-distances 1,2,4,8,16,32,64,128 \
  --fixed-anchor-distance 128 \
  --score-lift 0.25 \
  --singleton-top-n 16 \
  --singleton-ranking-metric abs_positive_suppression_x_dm_dscore \
  --prefill-chunk-size 64 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation eager \
  --original-max-position-embeddings 40960 \
  --seed-start 0 \
  --num-seeds 1 \
  --output-dir "$VALUE_OUT" \
  >"$VALUE_OUT/stdout.log" 2>"$VALUE_OUT/stderr.log" &
PID_VALUE=$!

MPR_VARIANTS="exact_pre_top2_postscore,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_masspreserve,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random_masspreserve,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_masspreserve,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random_masspreserve"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=7 \
TOKENIZERS_PARALLELISM=false \
PHASE_PREKEY_STORAGE=cuda \
"$PYTHON" "$ROOT/src/run_phase_coherent_rope_probe_8b.py" \
  --model-name-or-path "$MODEL" \
  --output-dir "$MPR_OUT" \
  --lengths 8192 \
  --seed-start 0 \
  --num-seeds 1 \
  --ratio 0.02 \
  --variants "$MPR_VARIANTS" \
  --local-window 128 \
  --sink-tokens 16 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 \
  --global-max-position 70000 \
  >"$MPR_OUT/run.log" 2>&1 &
PID_MPR=$!

STATUS=0
wait "$PID_VALUE" || STATUS=$?
wait "$PID_MPR" || STATUS=$?

if [[ "$STATUS" -eq 0 ]]; then
  printf 'complete\n' > "$QUEUE/COMPLETE"
else
  printf 'failed:%s\n' "$STATUS" > "$QUEUE/FAILED"
fi
exit "$STATUS"

