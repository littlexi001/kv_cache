# Design

## Falsifiable Conjecture

Post-hoc top-head pruning failed partly because Qwen3 was trained with dense
attention heads. If the model is trained from random initialization with a
hard routed-head constraint, it may learn to distribute token information
across only 4 selected heads per layer without immediate training collapse.

## Physical Prior

The previous experiments removed head-token links after the dense model had
already learned to use shared sink and recent tokens across many heads. That
intervention changed the model's learned computation. A model trained with the
constraint from step 0 may learn different head specialization and routing.

## Mathematical Model

For each layer and token representation `x[l, t]`, compute:

```text
gate_logits[l, t] = W_gate[l] x[l, t]
selected_heads[l, t] = top4(gate_logits[l, t])
```

The hard routing mask is:

```text
M[l, t, h] = 1 if h in selected_heads[l, t], else 0
```

Attention uses 16 query heads and 16 KV heads. For token `t`, only heads with
`M[l, t, h] = 1` receive nonzero Q/K/V. During attention, key token `j` is
visible to head `h` only if:

```text
j <= t and M[l, j, h] = 1
```

The output of head `h` for query token `t` is zeroed if:

```text
M[l, t, h] = 0
```

## Implementation Contract

The first implementation uses dense tensors and masks:

```text
K/V shape: [batch, 16 heads, seq, head_dim]
head-token mask shape: [batch, seq, 16]
```

This means it tests the training behavior of routed heads. It does not yet test
actual ragged KV cache memory savings.

## Gate Training

Forward routing is hard top4. Backward uses a straight-through estimator:

```text
route = hard_top4 + softmax(gate_logits) - stopgrad(softmax(gate_logits))
```

Auxiliary losses:

```text
router_load_loss: encourages average gate probability to be balanced over heads
router_z_loss: penalizes large router logits
```

The default total loss is:

```text
CE + 0.01 * router_load_loss + 0.001 * router_z_loss
```

## Experiment

Training data:

```text
/mnt/workspace/dclm/**/*.txt
```

The training script does not use a single fixed shard by default. Each run
samples files from the full DCLM directory tree, tokenizes the sampled text, and
stores the exact sampled file list in `token_cache/train_tokens_meta.json`.

Model config and tokenizer source:

```text
/mnt/workspace/Qwen3-0.6B
```

Default training setting:

```text
8 GPUs
seq_len = 2048
per_device_batch_size = 1
gradient_accumulation_steps = 8
max_train_seconds = 72000
```

## Pass Conditions

The first run is considered technically valid if:

1. training runs for hours without NaN;
2. CE loss decreases from the initial random baseline;
3. router load does not collapse to a small fixed set of heads;
4. checkpoints are saved and can resume;
5. TensorBoard records CE, auxiliary losses, entropy, and head load stats.

## Failure Conditions

The operationalization fails if:

1. attention produces NaNs because selected queries have no valid keys;
2. gate load collapses so most tokens use the same 4 heads;
3. CE loss does not decrease beyond random-init noise;
4. training is too slow to collect meaningful evidence in the 20-hour budget.

## Claim Boundary

A 20-hour run from random initialization cannot prove final language-model
quality. It can only test whether the routed-head architecture is trainable and
whether the gate avoids early collapse. A dense Qwen3 baseline trained for the
same token budget is still needed for a quality comparison.
