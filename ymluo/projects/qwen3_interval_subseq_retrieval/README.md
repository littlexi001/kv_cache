# Qwen3 Interval Subsequence Retrieval

This project trains a small fdong-style `MyQwen3ForCausalLM` on synthetic
next-token prediction data built from arithmetic subsequences.

The data is generated as token ids directly, not as strings passed through a
tokenizer.

## Data

Default settings:

```text
total_token = 10000
subseq_len = 4
seq_len = 1024
intervals = 1,2,3
```

For `interval=1`, the subsequence table is:

```text
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]
...
[9997, 9998, 9999, 10000]
```

For `interval=2`, the default scaled mode keeps 2500 candidate groups and scales
the token ids:

```text
[2, 4, 6, 8]
[10, 12, 14, 16]
...
```

With the default small setting, the maximum token id is `10000 * 3 = 30000`,
which fits the base Qwen vocab size `151936`.

Each training sample randomly samples 256 candidate groups and concatenates
them into a 1024-token sequence:

```text
sequence = 256 shuffled subsequences
```

Training is standard causal next-token prediction:

```text
loss = cross_entropy(model(sequence).logits[:, :-1], sequence[:, 1:])
```

All 1023 valid next-token positions contribute to the parameter update. There is
no query token, placeholder, or final answer-only objective in this project.

## Model

The script overrides the loaded Qwen config to use 8 layers by default:

```text
num_hidden_layers = 8
attention_stride_pattern = 1,1,4,4,4,4,1,1
```

You do not need to edit the original `config.json`. Editing
`num_hidden_layers` in a config file is fine for training a new 8-layer model,
but it will not be compatible with 28-layer checkpoints. This project handles
the override with command-line arguments so other experiments are not affected.

## Run

Single GPU:

```bash
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_train.sh
```

Nohup:

```bash
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/nohup_train.sh
```

Multi-GPU:

```bash
CUDA_DEVICES=0,1,2,3 bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_ddp_train.sh
```

Multi-GPU with `nohup`:

```bash
CUDA_DEVICES=0,1,2,3 bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/nohup_ddp_train.sh
```

Useful overrides:

```bash
INTERVALS=1,2,4 \
RUN_NAME=unet8-intervals-1-2-4 \
TOTAL_STEPS=20000 \
BATCH_SIZE=4 \
TRAIN_MODE=full_sequence_lm \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/nohup_train.sh
```

When `INTERVAL_GROUP_MODE=scaled`, the largest generated token id is
`TOTAL_TOKEN * max(INTERVALS)`. If that exceeds the base Qwen vocab, the trainer
automatically increases `config.vocab_size` unless `AUTO_RESIZE_VOCAB=false`.
For example, `TOTAL_TOKEN=100000` and `INTERVALS=1,2,3` needs vocab size
  `300001`. To keep using checkpoints with the base vocab, use the default
  small setting: `TOTAL_TOKEN=10000 INTERVALS=1,2,3`.

Use a different 8-layer U-Net schedule:

```bash
ATTENTION_STRIDE_PATTERN=1,4,4,8,8,4,4,1 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/run_train.sh
```

Outputs:

```text
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_interval_subseq_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
```

## Dump Attention Scores

After a checkpoint is saved, dump each layer/head attention values:

```bash
RUN_NAME=unet8-interval1-lm-ddp \
CKPT_STEP=2000 \
bash ymluo/projects/qwen3_interval_subseq_retrieval/scripts/dump_attention_scores.sh
```

By default this saves both raw masked attention scores and softmax
probabilities. Files are written under:

```text
outputs/train/<run_name>/attention_scores/step_<ckpt_step>/
```

The per-head tensor files have shape:

```text
[batch, selected_query_positions, key_positions]
```

Use `QUERY_POSITIONS=last` or `QUERY_POSITIONS=0,255,511,767,1023` if saving the
full `[1024,1024]` matrix for every layer/head is too large.
