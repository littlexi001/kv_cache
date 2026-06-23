# Design

## Falsifiable Conjecture

A gate trained on top of the official Qwen3-0.6B model can predict which KV
heads should store each token. If this is true, the model can reduce average
KV-cache token-head slots to about `20%` of the original cache while keeping
next-token CE from collapsing during continued training.

## Physical Prior

Earlier observation-window experiments showed that different tokens need
different numbers of heads. Sink and high-attention tokens need more coverage,
while many background tokens can use fewer heads. A learned gate may predict
this need from the token representation at write time, without observing future
queries.

## Mathematical Model

For layer `l`, token `t`, and hidden state `x[l,t]`:

```text
logits[l,t] = W_gate[l] x[l,t]
prob[l,t,h] = sigmoid(logits[l,t,h] / temperature)
hard[l,t,h] = selected by the hard gate
```

The default hard gate is `global_budget`:

```text
1. Sink tokens keep all KV heads.
2. Every non-sink token gets its highest-logit KV head.
3. Remaining token-head slots are filled by global top logits until
   target_keep_ratio is reached for the layer batch.
```

The target budget is:

```text
E_{l,t,h}[hard[l,t,h]] ~= target_keep_ratio
```

The default target is:

```text
target_keep_ratio = 0.20
```

For Qwen3-0.6B, `num_key_value_heads = 8`, so the target average is roughly
`1.6` KV heads per token.

## Implementation Contract

The first implementation patches Hugging Face Qwen3 eager attention:

```text
input hidden_states -> q_proj/k_proj/v_proj
hidden_states -> kv_gate
gate mask applies to K/V token-head slots
query heads inherit the mask of their corresponding KV head group
```

The implementation still stores dense K/V tensors during training. It measures
the routed-KV training behavior and expected KV-cache slot count. It does not
yet implement ragged cache allocation.

## Loss

```text
total_loss = CE
           + budget_loss_coef * budget_loss
           + load_loss_coef * load_balance_loss
           + z_loss_coef * gate_z_loss
```

Default coefficients:

```text
budget_loss_coef = 0.05
load_loss_coef = 0.01
z_loss_coef = 0.001
```

## Pass Conditions

The first run is useful if:

1. training runs without NaN;
2. CE loss does not collapse relative to the starting model;
3. mean hard keep ratio approaches `0.20`;
4. per-head load does not collapse to one or two KV heads;
5. checkpoints can resume;
6. DCLM streaming metadata shows the full data tree was discovered and sharded.

## Failure Conditions

The operationalization fails if:

1. mean keep ratio cannot approach `0.20`;
2. CE rises sharply and does not recover;
3. gate load collapses to a small fixed subset of KV heads;
4. training repeatedly reads one small fixed data shard;
5. the patched attention diverges from Qwen3 tensor shapes or cache behavior.

## Claim Boundary

This project trains a dense masked attention implementation. A successful run
does not prove real wall-clock speedup until a ragged KV-cache runtime is added.
It only tests whether the model can learn a token-to-KV-head write policy under
a 20% average KV budget.
