# Qwen1.5 MoE Real Attention Cluster

Train `/mnt/workspace/Qwen1.5-MoE-A2.7B` from random initialization on DCLM text
with the real-data version of the attention-cluster experiment.

## What This Changes

For each MoE decoder layer:

```text
input_layernorm(hidden)
-> q_proj/q_norm
-> q is used as the MoE gate input

full attention still runs with output_attentions=True
-> top 10% previous tokens by attention score are selected per query/head
-> their attention weights are renormalized
-> selected V vectors are weighted and projected
-> residual + sparse top-10% attention output
-> post_attention_layernorm
-> MoE experts
```

The training loss is:

```text
L = L_lm
  + attention_cluster_weight * L_attention_cluster
  + load_balance_loss_weight * L_load_balance
```

`L_attention_cluster` uses the same pairwise idea as the synthetic demo:
tokens that attend to the same top 10% history neighborhood are encouraged to
have overlapping router distributions.

## Data

The dataset reader recursively discovers `*.txt` files under `/mnt/workspace/dclm`.
It does not concatenate or count the full 4T corpus. Each worker repeatedly:

1. randomly selects one text file;
2. randomly seeks to a byte offset;
3. reads the next line;
4. tokenizes it and yields fixed-length token blocks.

This samples approximately by bytes and keeps memory bounded.

## Run

```bash
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

Main defaults:

```text
MODEL_PATH=/mnt/workspace/Qwen1.5-MoE-A2.7B
DATA_PATH=/mnt/workspace/dclm
NPROC_PER_NODE=8
SEQ_LENGTH=1024
ATTENTION_TOP_RATIO=0.10
EXPERT_INPUT_TOP_RATIO=0.10
ATTENTION_CLUSTER_WEIGHT=0.01
LOAD_BALANCE_LOSS_WEIGHT=0.01
INIT_FROM_SCRATCH=true
```

If the full model does not fit with plain DDP, enable ZeRO-3:

```bash
DEEPSPEED_CONFIG=ymluo/projects/qwen15_moe_real_attention_cluster/configs/deepspeed_zero3.json \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

## Resume

Resume the latest checkpoint under the output directory:

```bash
RESUME_FROM_CHECKPOINT=auto \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

Resume a specific checkpoint:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-5000 \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

Checkpoints are standard Hugging Face Trainer checkpoints and include optimizer
and scheduler state.
