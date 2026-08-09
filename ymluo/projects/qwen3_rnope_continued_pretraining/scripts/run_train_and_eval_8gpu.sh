#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/fdong/ymluo/projects/qwen3_rnope_continued_pretraining
INFER=/home/fdong/ymluo/projects/qwen3_inference_rnope
PY=/home/fdong/miniconda3/envs/moe/bin/python
TORCHRUN=/home/fdong/miniconda3/envs/moe/bin/torchrun
MODEL=/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
TOKENS=/home/fdong/ymluo/projects/parallel_block_retrieval/data/pg19_10m_continuation_memory_v1/base_blocks.npy
RULER=/home/fdong/ymluo/projects/qwen3_ruler32k_rope_method/data/qwen3_8b_ruler13_32k_m2_seed42.jsonl
RUN_NAME=${1:-rnope_offset3_pg19_10m_20260805}
RUN=$ROOT/outputs/$RUN_NAME
mkdir -p "$RUN"

on_exit() {
  rc=$?
  if test "$rc" -ne 0; then touch "$RUN/launcher.failed"; fi
}
trap on_exit EXIT

run_train() {
  local seq=$1
  local steps=$2
  local out=$3
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$TORCHRUN" --standalone --nproc_per_node=8 \
    "$ROOT/src/train_rnope_qlora.py" \
    --model-name-or-path "$MODEL" \
    --token-blocks "$TOKENS" \
    --output-dir "$out" \
    --sequence-length "$seq" \
    --max-steps "$steps" \
    --eval-sequences 32 \
    --learning-rate 2e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --save-steps 40
}

if run_train 8192 1 "$RUN/smoke_8k" >"$RUN/smoke_8k.log" 2>&1; then
  SEQ=8192
  STEPS=153
  echo 8192 >"$RUN/selected_sequence_length.txt"
else
  touch "$RUN/smoke_8k.failed"
  run_train 4096 1 "$RUN/smoke_4k" >"$RUN/smoke_4k.log" 2>&1
  SEQ=4096
  STEPS=306
  echo 4096 >"$RUN/selected_sequence_length.txt"
fi
touch "$RUN/smoke.done"

run_train "$SEQ" "$STEPS" "$RUN/train" >"$RUN/train.log" 2>&1
test -f "$RUN/train/train.done"
touch "$RUN/training.done"

pids=()
for shard in $(seq 0 7); do
  out="$RUN/ruler/shard$shard"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$shard" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$INFER/src/run_inference_rnope_ruler.py" \
    --model-name-or-path "$MODEL" \
    --adapter-path "$RUN/train/final_adapter" \
    --examples-jsonl "$RULER" \
    --output-dir "$out" \
    --variants native_rope,nope_every4_offset3 \
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

"$PY" "$INFER/src/summarize_inference_rnope.py" --run-dir "$RUN/ruler" \
  >"$RUN/ruler/summary_stdout.log" 2>"$RUN/ruler/summary_stderr.log"
touch "$RUN/launcher.done"
trap - EXIT
