# Qwen1.5 MoE Real Attention Cluster

Train a resized Qwen1.5-MoE-style model from random initialization on DCLM text
with the real-data version of the attention-cluster experiment. The default run
uses the tokenizer/model class from `/mnt/workspace/Qwen1.5-MoE-A2.7B`, but
shrinks the config to a roughly 0.6B-parameter MoE before initialization.

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
EXPERIMENT_MODE=attention_cluster
MODEL_SIZE_PRESET=moe_0_6b
NPROC_PER_NODE=8
SEQ_LENGTH=1024
ATTENTION_TOP_RATIO=0.10
EXPERT_INPUT_TOP_RATIO=0.10
ATTENTION_CLUSTER_WEIGHT=0.01
LOAD_BALANCE_LOSS_WEIGHT=0.01
INIT_FROM_SCRATCH=true
GRADIENT_ACCUMULATION_STEPS=4
DEEPSPEED_CONFIG=
```

To run an unpatched MoE baseline with the same 0.6B model size and training
hyperparameters:

```bash
EXPERIMENT_MODE=baseline \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

Baseline mode skips the attention-cluster patch completely: no gate-input
replacement, no sparse expert attention replacement, and no forced
`output_attentions=True`.

The `moe_0_6b` preset applies these config overrides before random
initialization:

```text
hidden_size=768
intermediate_size=2048
moe_intermediate_size=1024
shared_expert_intermediate_size=1024
num_hidden_layers=12
num_attention_heads=12
num_key_value_heads=4
num_experts=12
num_experts_per_tok=2
decoder_sparse_step=1
mlp_only_layers=[]
```

This smaller model fits without ZeRO-3 on typical 8-GPU training nodes. To run
the original full model config, use:

```bash
MODEL_SIZE_PRESET=none \
DEEPSPEED_CONFIG=ymluo/projects/qwen15_moe_real_attention_cluster/configs/deepspeed_zero3.json \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

To keep the 0.6B preset but change a few dimensions, pass JSON config
overrides:

```bash
MODEL_CONFIG_OVERRIDES='{"num_hidden_layers": 8, "num_experts": 8}' \
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
