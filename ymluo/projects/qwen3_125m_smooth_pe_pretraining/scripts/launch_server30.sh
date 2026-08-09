#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/fdong/ymluo/projects/qwen3_125m_smooth_pe_pretraining
PYTHON=/home/fdong/miniconda3/envs/moe/bin/python
mkdir -p "$PROJECT/outputs/deep_highfreq_drop_v2" "$PROJECT/outputs/slow_rope_v2"

CUDA_VISIBLE_DEVICES=4,5 nohup "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29630 \
  "$PROJECT/src/train_synthetic_pe.py" \
  --variant deep_highfreq_drop \
  --output-dir "$PROJECT/outputs/deep_highfreq_drop_v2" \
  --tokens 20000000 --sequence-length 2048 --micro-batch 2 --grad-accum 4 \
  > "$PROJECT/outputs/deep_highfreq_drop_v2/launcher.log" 2>&1 < /dev/null &
echo $! > "$PROJECT/outputs/deep_highfreq_drop_v2/launcher.pid"

CUDA_VISIBLE_DEVICES=6,7 nohup "$PYTHON" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29631 \
  "$PROJECT/src/train_synthetic_pe.py" \
  --variant slow_rope \
  --output-dir "$PROJECT/outputs/slow_rope_v2" \
  --tokens 20000000 --sequence-length 2048 --micro-batch 2 --grad-accum 4 \
  > "$PROJECT/outputs/slow_rope_v2/launcher.log" 2>&1 < /dev/null &
echo $! > "$PROJECT/outputs/slow_rope_v2/launcher.pid"

echo "server30 jobs launched"
