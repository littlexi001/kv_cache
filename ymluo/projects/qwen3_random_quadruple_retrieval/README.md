# Qwen3 Random Quadruple Retrieval

This project is a data-generation variant of
`qwen3_interval_subseq_retrieval`. It keeps the same direct-token causal LM
training path, but replaces arithmetic interval subsequences with a fixed file
of random quadruples.

## Data

Default settings:

```text
token ids = 1..1000
quadruple_len = 4
num_quadruples = 100000
seq_len = 1024
quadruples_per_sequence = 256
```

The quadruple table is saved as a tensor file:

```text
ymluo/projects/qwen3_random_quadruple_retrieval/data/random_quadruples_1000_100000.pt
```

Each row is sampled independently from token ids `1..1000`, so examples look
like:

```text
[37, 812, 4, 456]
[1000, 19, 271, 271]
...
```

Each training sample randomly selects 256 rows from the table and concatenates
them into a 1024-token sequence:

```text
sequence = 256 random quadruples
```

Training is standard causal next-token prediction:

```text
loss = cross_entropy(model(sequence).logits[:, :-1], sequence[:, 1:])
```

All 1023 valid next-token positions contribute to the parameter update. There is
no query token, placeholder, or final answer-only objective.

## Generate Quadruples

The training script creates the quadruple file automatically if it is missing.
You can also create it explicitly:

```bash
bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/generate_quadruples.sh
```

Use a different seed or overwrite the existing file:

```bash
QUADRUPLE_SEED=42 FORCE=true \
bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/generate_quadruples.sh
```

## Model

The script follows the interval-subsequence project and loads Qwen3 config from:

```text
/mnt/workspace/Qwen3-0.6B
```

By default it also keeps the same 8-layer override:

```text
num_hidden_layers = 8
attention_stride_pattern = 1,1,4,4,4,4,1,1
```

Override these arguments if you want to train a different layer count.

## Run

Single GPU:

```bash
bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/run_train.sh
```

Nohup:

```bash
bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/nohup_train.sh
```

Multi-GPU:

```bash
CUDA_DEVICES=0,1,2,3 bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/run_ddp_train.sh
```

Multi-GPU with `nohup`:

```bash
CUDA_DEVICES=0,1,2,3 bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/nohup_ddp_train.sh
```

Useful overrides:

```bash
RUN_NAME=unet8-random-quad-lm \
TOTAL_STEPS=20000 \
BATCH_SIZE=4 \
QUADRUPLE_SEED=20260518 \
bash ymluo/projects/qwen3_random_quadruple_retrieval/scripts/nohup_train.sh
```

Outputs:

```text
ymluo/projects/qwen3_random_quadruple_retrieval/outputs/train/<run_name>/metrics.jsonl
ymluo/projects/qwen3_random_quadruple_retrieval/outputs/train/<run_name>/checkpoints/<step>.pth
ymluo/projects/qwen3_random_quadruple_retrieval/outputs/train/<run_name>/checkpoints/runtime_config.json
```
