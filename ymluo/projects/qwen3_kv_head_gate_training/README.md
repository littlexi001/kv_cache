# Qwen3 KV-Head Gate Training

This project starts from the official Qwen3-0.6B checkpoint and adds one
trainable gate per attention layer. The gate decides which KV heads should store
each token. The target is to reduce average KV-cache token-head slots to about
`20%` of the original full KV cache.

The project is different from observation-window assignment:

```text
observation-window assignment:
  observe future queries, then infer a token-head assignment

this project:
  token arrives -> gate(hidden_state) -> KV heads to write
```

## Model Change

Qwen3-0.6B uses GQA:

```text
num_attention_heads = 16
num_key_value_heads = 8
```

The gate is applied to KV heads, not query heads:

```text
gate_logits[l, t] = W_gate[l] x[l, t]
gate_prob[l, t] = sigmoid(gate_logits[l, t] / temperature)
hard_keep[l, t, kv_head] = selected by the hard gate
```

Default hard gate mode is `global_budget`: each layer first protects sink
tokens, then gives each non-sink token at least one KV head, then fills the
remaining highest-logit token-head slots until the current average KV budget is
reached.

The training budget uses a curriculum by default:

```text
initial_keep_ratio = 0.50
target_keep_ratio = 0.20
keep_ratio_anneal_steps = 30000
```

So the final target is still 20% KV-cache slots, but the model is not forced
into the 20% hard constraint at the first step.

The default gate is now a small MLP:

```text
LayerNorm(hidden) -> Linear(hidden, 256) -> SiLU -> Linear(256, kv_heads)
```

Set `GATE_TYPE=linear` when resuming an old checkpoint that was trained with
the original linear gate.

During training, the dense attention tensor is still used. Attention
probabilities are multiplied by a straight-through hard gate and renormalized.
This gives the hard routing behavior in the forward pass while still allowing
the gate to receive useful gradients from CE. Recent query-key pairs are
protected by default:

```text
gate_recent_tokens_all_heads = 256
```

This approximates a runtime cache where the most recent tokens remain dense and
older tokens are routed. The project still does not implement a ragged KV-cache
storage kernel.

## Loss

The default training loss is:

```text
total_loss = CE
           + budget_loss_coef * budget_loss
           + load_loss_coef * load_balance_loss
           + z_loss_coef * gate_z_loss
```

where:

```text
budget_loss = ((mean_gate_prob - target_keep_ratio) / target_keep_ratio)^2
load_balance_loss = mean_h ((mean_gate_prob_h - target_keep_ratio)^2)
gate_z_loss = mean(logsumexp(gate_logits)^2)
```

Default target:

```text
target_keep_ratio = 0.20
```

With 8 KV heads, this means the average target is about `1.6` KV heads per
token.

## Training Data

Default data is streamed from the full DCLM tree:

```text
/mnt/workspace/dclm/**/*.txt
```

The training script recursively discovers all matching text files, shards them
across DDP ranks, shuffles each rank's file order, and reads text chunks
sequentially. This avoids repeatedly training on one small fixed shard.

Streaming metadata is written to:

```text
<output_dir>/streaming_data_meta.json
```

## Run On Server

Start 8-GPU training:

```bash
cd /mnt/workspace/lym_code/scripts/kv_cache/kv_cache/ymluo/projects/qwen3_kv_head_gate_training
bash scripts/nohup_train_8x80g.sh
```

Defaults:

```text
model_name_or_path = /mnt/workspace/Qwen3-0.6B
train_data_root = /mnt/workspace/dclm
seq_len = 2048
per_device_batch_size = 1
gradient_accumulation_steps = 8
max_train_seconds = 72000
target_keep_ratio = 0.20
initial_keep_ratio = 0.50
keep_ratio_anneal_steps = 30000
gate_type = mlp
gate_hidden_size = 256
gate_recent_tokens_all_heads = 256
gate_hard_mode = global_budget
train_base_model = true
```

Resume:

```bash
RUN_DIR=/path/to/run
OUTPUT_DIR="${RUN_DIR}" RESUME_FROM="${RUN_DIR}/latest_checkpoint" bash scripts/nohup_train_8x80g.sh
```

TensorBoard:

```bash
bash scripts/tensorboard.sh
```

## Outputs

Each run writes:

```text
args.json
streaming_data_meta.json
tokenizer/
tensorboard/
checkpoint-0000500/
latest_checkpoint
```

Each checkpoint contains:

```text
model_state.pt
gate_state.pt
optimizer.pt
trainer_state.json
```

## Smoke Test

Local syntax and tiny-module smoke test:

```bash
python src/train_kv_head_gate_qwen3.py --smoke_test
```
