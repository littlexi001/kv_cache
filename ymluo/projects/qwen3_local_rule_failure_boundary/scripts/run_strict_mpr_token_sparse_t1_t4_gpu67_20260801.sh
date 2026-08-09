#!/usr/bin/env bash
set -euo pipefail

# Frozen exact-pre support + genuinely token-sparse strict MPR.
# This launcher is intentionally hard-coded to the user-authorized GPUs 6-7.
# It is a launcher artifact only: creating/testing it does not start GPU work.
#
# For each layer and query head, the exact baseline freezes:
#   (1) exact-pre 2% selected support,
#   (2) the uncapped positive-gap (>1 QK logit) raw trigger mask,
#   (3) deterministic top-1 and top-4 token-capped masks,
#   (4) the desired score lift for every selected token.
# Every treatment/control arm replays those tensors.  Random-frequency arms
# additionally replay the corresponding plain arm's actual plane count and L2.

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$ROOT/outputs/20260801_strict_mpr_token_sparse_t1_t4_gpu67"
VARIANTS="exact_pre_top2_postscore,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_masspreserve,strict_mpr_pre_w128_lift25_gap1_t1_f8_cap0p25_random_masspreserve,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_masspreserve,strict_mpr_pre_w128_lift25_gap1_t4_f8_cap0p25_random_masspreserve"

mkdir -p "$OUT/gpu6" "$OUT/gpu7"

run_shard() {
  local physical_gpu="$1"
  local seed_start="$2"
  local shard="$3"
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
    TOKENIZERS_PARALLELISM=false \
    PHASE_PREKEY_STORAGE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" "$ROOT/src/run_phase_coherent_rope_probe_8b.py" \
      --model-name-or-path "$MODEL" \
      --output-dir "$OUT/$shard" \
      --lengths 8192,32768 \
      --seed-start "$seed_start" \
      --num-seeds 2 \
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
      > "$OUT/$shard/run.log" 2>&1
}

run_shard 6 0 gpu6 &
pid6=$!
run_shard 7 2 gpu7 &
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
