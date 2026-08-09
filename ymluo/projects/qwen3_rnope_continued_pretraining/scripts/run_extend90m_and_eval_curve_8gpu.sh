#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_rnope_continued_pretraining
INFER=/home/fdong/ymluo/projects/qwen3_inference_rnope
PY=/home/fdong/miniconda3/envs/moe/bin/python
TORCHRUN=/home/fdong/miniconda3/envs/moe/bin/torchrun
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TOKENS=/home/fdong/ymluo/projects/parallel_block_retrieval/data/pg19_10m_continuation_memory_v1/base_blocks.npy
RULER=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
INITIAL=/home/fdong/ymluo/projects/qwen3_rnope_continued_pretraining/outputs/rnope_offset3_pg19_10m_20260805_v2/train/final_adapter
RUN_NAME=${1:-rnope_offset3_pg19_cumulative100m_20260805}
RUN=$ROOT/outputs/$RUN_NAME
mkdir -p "$RUN"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

if ! test -f "$RUN/resume_smoke.done"; then
  rm -rf "$RUN/resume_smoke"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
    "$ROOT/src/train_rnope_qlora.py" \
    --model-name-or-path "$MODEL" \
    --adapter-path "$INITIAL" \
    --token-blocks "$TOKENS" \
    --output-dir "$RUN/resume_smoke" \
    --sequence-length 4096 \
    --max-steps 1 \
    --eval-sequences 8 \
    --learning-rate 5e-5 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --save-steps 1 \
    >"$RUN/resume_smoke.log" 2>&1
  test -f "$RUN/resume_smoke/train.done"
  touch "$RUN/resume_smoke.done"
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node=8 \
  "$ROOT/src/train_rnope_qlora.py" \
  --model-name-or-path "$MODEL" \
  --adapter-path "$INITIAL" \
  --token-blocks "$TOKENS" \
  --output-dir "$RUN/train" \
  --sequence-length 4096 \
  --max-steps 2747 \
  --eval-sequences 32 \
  --learning-rate 5e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --save-steps 305 \
  >"$RUN/train.log" 2>&1
test -f "$RUN/train/train.done"
touch "$RUN/training.done"

evaluate_adapter() {
  local adapter=$1
  local outroot=$2
  local variants=$3
  local pids=()
  for shard in $(seq 0 7); do
    local out="$outroot/shard$shard"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$shard" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" -u "$INFER/src/run_inference_rnope_ruler.py" \
      --model-name-or-path "$MODEL" \
      --adapter-path "$adapter" \
      --examples-jsonl "$RULER" \
      --output-dir "$out" \
      --variants "$variants" \
      --target-length 32768 \
      --max-samples-per-task 2 \
      --max-new-tokens-cap 128 \
      --prefill-chunk-size 256 \
      --dtype bfloat16 \
      --attn-implementation sdpa \
      --load-in-4bit \
      --shard-count 8 \
      --shard-index "$shard" \
      >"$out/stdout.log" 2>"$out/stderr.log" &
    echo $! >"$out/pid.txt"
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  "$PY" "$INFER/src/summarize_inference_rnope.py" --run-dir "$outroot" \
    >"$outroot/summary_stdout.log" 2>"$outroot/summary_stderr.log"
}

for adapter in "$RUN"/train/checkpoints/step_*; do
  step=$(basename "$adapter")
  out="$RUN/checkpoint_evals/$step"
  evaluate_adapter "$adapter" "$out" "nope_every4_offset3"
done

evaluate_adapter "$RUN/train/final_adapter" "$RUN/final_eval" \
  "native_rope,nope_every4_offset3"

"$PY" "$ROOT/src/summarize_training_curve.py" --run-dir "$RUN" \
  >"$RUN/training_curve_stdout.log" 2>"$RUN/training_curve_stderr.log"
touch "$RUN/launcher.done"
trap - EXIT
