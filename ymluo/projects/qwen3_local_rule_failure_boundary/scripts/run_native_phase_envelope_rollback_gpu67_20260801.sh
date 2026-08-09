#!/usr/bin/env bash
set -euo pipefail

# Frozen Qwen3-8B Native Phase Envelope + Coherent Rollback experiment.
# This launcher is intentionally restricted to the user-authorized GPUs 6-7.
# It only defines the job; creating this file does not start server work.

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$ROOT/outputs/20260801_native_phase_envelope_rollback_gpu67_v2"
VARIANTS="full_rope,rope_top2,exact_pre_top2_postscore,npe_native_pre_top2,npe_distance_clip_pre_top2,npe_rollback_pre_top2,npe_rollback_masspreserve_pre_top2,npe_random_matched_pre_top2,npe_frozen_rollback_pre_top2,npe_frozen_rollback_masspreserve_pre_top2,npe_frozen_random_matched_pre_top2"

mkdir -p "$OUT/gpu6" "$OUT/gpu7"

run_shard() {
  local gpu="$1"
  local seed_start="$2"
  local shard="$3"
  # Exact pre-RoPE keys add several GiB at 64K.  Keep that diagnostic cache on
  # host memory so the two 24-GiB RTX 3090s cannot silently OOM at the longest
  # context; the model/KV cache and all score computation remain on the GPU.
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false PHASE_PREKEY_STORAGE=cpu \
    "$PYTHON" "$ROOT/src/run_native_phase_envelope_rollback_8b.py" \
      --model-name-or-path "$MODEL" \
      --output-dir "$OUT/$shard" \
      --lengths 8192,32768,65536 \
      --seed-start "$seed_start" \
      --num-seeds 4 \
      --ratio 0.02 \
      --variants "$VARIANTS" \
      --local-window 128 \
      --sink-tokens 16 \
      --prefill-chunk-size 128 \
      --dtype bfloat16 \
      --load-in-4bit \
      --attn-implementation sdpa \
      --original-max-position-embeddings 40960 \
      --global-max-position 70000 \
      --npe-anchor-distances 0,1,2,4,8,16,32,64,128 \
      --npe-mad-lambda 2.5 \
      --npe-dense-rollback-tokens 64 \
      --npe-coarse-search-points 48 \
      --npe-refinement-steps 2 \
      --npe-refinement-bins 8 \
      --npe-reconstruction-guard-multiplier 2.0 \
      --npe-reconstruction-guard-floor 0.001 \
      > "$OUT/$shard/run.log" 2>&1
}

run_shard 6 0 gpu6 &
pid6=$!
run_shard 7 4 gpu7 &
pid7=$!

status=0
wait "$pid6" || status=$?
wait "$pid7" || status=$?

if [[ "$status" -eq 0 ]]; then
  printf 'ok\n' > "$OUT/done.txt"
else
  printf 'failed: %s\n' "$status" > "$OUT/failed.txt"
fi
exit "$status"
