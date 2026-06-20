# Qwen3 Routed Top4 MHA Pretraining

This project trains a randomly initialized Qwen3-style model where each token
is routed to only 4 of 16 attention heads in every layer.

The goal is to test whether the failure of post-hoc head-token pruning comes
from the dense-attention training format. This model is trained with routing
from the start.

## Model Change

Base configuration is read from:

```text
/mnt/workspace/Qwen3-0.6B/config.json
```

The script changes the attention to MHA:

```text
num_attention_heads = 16
num_key_value_heads = 16
head_dim = 128
hidden_size = 1024
```

Each layer has an independent gate:

```text
gate_logits = W_gate x
selected_heads = top4(gate_logits)
```

Forward pass uses hard top4 routing. Backward pass uses a straight-through
softmax estimator:

```text
route = hard_top4 + softmax(gate_logits) - stopgrad(softmax(gate_logits))
```

The implementation keeps dense head slots:

```text
K/V tensor shape remains [batch, 16 heads, seq, head_dim]
unselected token/head slots are masked
unselected query-head outputs are zeroed
```

This is not yet a ragged KV cache implementation. It is the first training
test for the routed-head architecture.

## Auxiliary Loss

Training loss:

```text
total_loss = CE
           + router_aux_loss_coef * router_load_loss
           + router_z_loss_coef * router_z_loss
```

Default coefficients:

```text
router_aux_loss_coef = 0.01
router_z_loss_coef = 0.001
```

TensorBoard logs:

- `train/ce_loss`
- `train/loss`
- `router/load_loss`
- `router/z_loss`
- `router/entropy`
- `router/hard_load_min`
- `router/hard_load_max`
- `router/hard_load_mean`

## Run On Server

Start 8-GPU training with nohup:

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
bash scripts/nohup_train_8x80g.sh
```

The default script uses:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
seq_len=2048
per_device_batch_size=1
gradient_accumulation_steps=8
max_train_seconds=72000
save_steps=500
```

Override examples:

```bash
RUN_NAME=test_longer bash scripts/nohup_train_8x80g.sh --seq_len 4096 --gradient_accumulation_steps 4
```

Use a different DCLM file sample for another run:

```bash
DATASET_SAMPLE_SEED=20260620 DATASET_SAMPLE_FILES=2048 bash scripts/nohup_train_8x80g.sh
```

If the first run spends a long time building the token cache, non-rank0
processes wait by polling the cache files instead of entering an NCCL barrier.
The default wait limit is 24 hours:

```bash
CACHE_WAIT_TIMEOUT_SECONDS=172800 bash scripts/nohup_train_8x80g.sh
```

Resume:

```bash
RUN_NAME=resume_test bash scripts/nohup_train_8x80g.sh \
  --resume_from /mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs/<run>/latest_checkpoint
```

Start TensorBoard:

```bash
bash scripts/tensorboard.sh
```

Default TensorBoard URL:

```text
http://<server-ip>:6006
```

## Outputs

Default output root:

```text
/mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_routed_top4_mha_pretrain/output/routed_top4_qwen3_0p6b_runs
```

Default training data:

```text
/mnt/workspace/dclm/**/*.txt
```

Each run samples files before building the token cache:

```text
dataset_sample_files = 1024
tokenize_max_chars = 200000000
tokenize_max_chars_per_file = 250000
```

This means the old `part-00000.txt` path is no longer the default training set.
It remains available only as a compatibility option through `--train_text_path`
when `--train_data_root ""` is also passed.

Default token cache:

```text
<output_dir>/token_cache
```

Each run contains:

```text
args.json
routed_qwen_config.json
token_cache/train_tokens.uint32.bin
token_cache/train_tokens_meta.json
tokenizer/
tensorboard/
checkpoint-0000500/
latest_checkpoint
```

`train_tokens_meta.json` records the discovered file count, sampled file count,
sample seed, character limits, token count, and the exact sampled file list.

## Smoke Test

Local tiny model smoke test:

```bash
python src/train_routed_top4_qwen.py --smoke_test
```

This verifies forward/backward with a tiny routed model. It does not load the
0.6B model.

## Downstream Eval

Prepare small multiple-choice validation sets:

```bash
bash scripts/prepare_downstream_eval_data.sh
```

Compare the latest routed checkpoint with the official Qwen3-0.6B model:

```bash
bash scripts/eval_checkpoint_vs_baseline.sh
```

Compare a specific checkpoint:

```bash
CHECKPOINT_DIR=/path/to/checkpoint-0008500 bash scripts/eval_checkpoint_vs_baseline.sh
```

The detailed method is documented in:

```text
docs/downstream_eval.md
```

Held-out non-DCLM text PPL:

```bash
bash scripts/prepare_heldout_ppl_text.sh
bash scripts/eval_heldout_ppl_vs_baseline.sh
```

This defaults to WikiText-103 validation text and compares the routed checkpoint
with `/mnt/workspace/Qwen3-0.6B`.
