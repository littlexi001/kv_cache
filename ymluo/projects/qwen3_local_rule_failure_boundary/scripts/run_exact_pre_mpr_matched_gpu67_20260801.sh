#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
OUT="$ROOT/outputs/20260801_exact_pre_mpr_matched_gpu67"
VARIANTS="full_rope,rope_top2,local_global_postscore,exact_pre_top2_postscore,exact_pre_top2_blend25,exact_dual_top2_postscore,clipped128_top2_postscore,remote_nope_cal_full,mpr_pre_w128_lift25,mpr_pre_w4k_lift25,mpr_pre_w4k_lift25_gap1,mpr_pre_w4k_lift25_masspreserve,mpr_pre_w128_lift25_gap1_f8,mpr_pre_w128_lift25_gap1_f16"

mkdir -p "$OUT/gpu6" "$OUT/gpu7"

run_shard() {
  local gpu="$1"
  local seed_start="$2"
  local shard="$3"
  CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" "$ROOT/src/run_phase_coherent_rope_probe_8b.py" \
      --model-name-or-path "$MODEL" \
      --output-dir "$OUT/$shard" \
      --lengths 8192,16384,32768 \
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
