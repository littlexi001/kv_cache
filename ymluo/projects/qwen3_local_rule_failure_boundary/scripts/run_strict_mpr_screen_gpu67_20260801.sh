#!/usr/bin/env bash
set -euo pipefail

# Small frozen-Qwen3-8B strict-MPR screen.  This launcher is intentionally
# restricted to the user-authorized GPUs 6-7.  Creating it does not start work.
#
# Fair controls use identical prompts, seeds, 2% token budget, and local/sink
# reservation.  The exact baseline must appear first: it freezes each layer's
# token support, trigger mask, and numeric target plan for all later controls.
#   exact_pre_top2_postscore                         capture + selector only
#   strict...cap0p25                                heuristic-support rescue
#   strict...cap0p25_random                         cross-arm count/L2-matched planes
#   strict...cap0p25_masspreserve                   partition-preserving ablation
#   strict...cap0p25_random_masspreserve            random + partition control

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$ROOT/outputs/20260801_strict_mpr_frozen_screen_gpu67"
VARIANTS="full_rope,rope_top2,exact_pre_top2_postscore,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_random,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_masspreserve,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_random_masspreserve"

mkdir -p "$OUT/gpu6" "$OUT/gpu7"

run_shard() {
  local gpu="$1"
  local seed_start="$2"
  local shard="$3"
  CUDA_VISIBLE_DEVICES="$gpu" \
    TOKENIZERS_PARALLELISM=false \
    PHASE_PREKEY_STORAGE=cuda \
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
