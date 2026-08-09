#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUNNER="$PROJECT/src/run_phase_rope_local_order_control_8b.py"
ROOT="$PROJECT/outputs/20260801_phase_rope_local_order_matched_methods_gpu67"
VARIANTS=full_rope,rope_top2,exact_pre_top2_postscore,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25,strict_mpr_pre_w128_lift25_gap1_f8_cap0p25_masspreserve,npe_native_pre_top2,npe_rollback_pre_top2,npe_rollback_masspreserve_pre_top2

mkdir -p "$ROOT/local_gpu6" "$ROOT/remote_gpu7"

# GPU 6: same-word-bag counterfactual pairs; the evidence is always local.
CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false PHASE_PREKEY_STORAGE=cuda \
  "$PYTHON" "$RUNNER" \
  --model-name-or-path "$MODEL" \
  --output-dir "$ROOT/local_gpu6" \
  --task-families local_order \
  --local-lengths 8192,32768 \
  --remote-lengths 8192 \
  --seed-start 0 \
  --num-seeds 2 \
  --variants "$VARIANTS" \
  --ratio 0.02 \
  --local-window 128 \
  --sink-tokens 16 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 \
  --global-max-position 70000 \
  >"$ROOT/local_gpu6.log" 2>&1 &
PID_LOCAL=$!

# GPU 7: independent seeds of the controlled remote two-hop task.
CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false PHASE_PREKEY_STORAGE=cuda \
  "$PYTHON" "$RUNNER" \
  --model-name-or-path "$MODEL" \
  --output-dir "$ROOT/remote_gpu7" \
  --task-families remote_retrieval \
  --local-lengths 8192 \
  --remote-lengths 8192,32768 \
  --seed-start 0 \
  --num-seeds 4 \
  --variants "$VARIANTS" \
  --ratio 0.02 \
  --local-window 128 \
  --sink-tokens 16 \
  --prefill-chunk-size 128 \
  --dtype bfloat16 \
  --load-in-4bit \
  --attn-implementation sdpa \
  --original-max-position-embeddings 40960 \
  --global-max-position 70000 \
  >"$ROOT/remote_gpu7.log" 2>&1 &
PID_REMOTE=$!

STATUS=0
wait "$PID_LOCAL" || STATUS=$?
wait "$PID_REMOTE" || STATUS=$?
if [[ "$STATUS" -eq 0 ]]; then
  printf 'ok\n' > "$ROOT/done.txt"
else
  printf 'failed: %s\n' "$STATUS" > "$ROOT/failed.txt"
fi
exit "$STATUS"
