#!/usr/bin/env bash
set -euo pipefail

EXP=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining/experiments/natural95_synth5_v1
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
RAW=/home/fdong/data/openweb_every_4096
TRAIN_BIN="$EXP/data/train_2p5b.uint16"
TARGET_TOKENS=2500000000

ensure_data() {
  if "$PYTHON" - "$TRAIN_BIN" "$TARGET_TOKENS" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
target = int(sys.argv[2])
meta = path.with_suffix(path.suffix + ".meta.json")
ok = path.is_file() and meta.is_file()
if ok:
    payload = json.loads(meta.read_text())
    ok = payload.get("token_count") == target and path.stat().st_size == target * 2
raise SystemExit(0 if ok else 1)
PY
  then
    echo "verified existing $TRAIN_BIN"
    return
  fi

  "$PYTHON" "$EXP/src/prepare_natural_corpus.py" build-bin \
    --input-root "$RAW" \
    --pattern 'openweb_every_4096_*.txt' \
    --tokenizer "$EXP/data/tokenizer.json" \
    --output "$TRAIN_BIN" \
    --target-tokens "$TARGET_TOKENS"
}

launch() {
  local variant=$1 gpu_list=$2 port=$3
  local output="$EXP/outputs/scale2p5b_${variant}"
  if [[ -e "$output/config.json" || -e "$output/DONE" ]]; then
    echo "refusing to reuse $output" >&2
    return 1
  fi
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu_list" OMP_NUM_THREADS=1 "$PYTHON" -m torch.distributed.run \
    --nproc_per_node=4 --master_port="$port" "$EXP/src/train_mixed_pe.py" \
    --variant "$variant" --output-dir "$output" \
    --train-bin "$TRAIN_BIN" \
    --validation-bin "$EXP/data/validation_16m.uint16" \
    --tokenizer-meta "$EXP/data/tokenizer.meta.json" \
    --tokens "$TARGET_TOKENS" --sequence-length 2048 --micro-batch 2 --grad-accum 2 \
    --synthetic-fraction 0.05 --training-queries 32 --answer-weight 16 \
    --learning-rate 3e-4 --warmup-steps 1000 --log-every 100 \
    --eval-every 5000 --save-every 5000 --eval-samples 32 --natural-eval-samples 32 \
    > "$output/launcher.log" 2>&1 &
  echo $! > "$output/launcher.pid"
  echo "$variant pid=$(cat "$output/launcher.pid") GPUs=$gpu_list"
}

ensure_data
launch native 0,1,2,3 29860
launch complementary_smooth 4,5,6,7 29861
wait
