# Experiment Design

## Objective

Train a KV-head gate on top of official Qwen3-0.6B and test whether a gentler
curriculum can drive average long-term KV-head usage toward `20%` without the
CE collapse seen when the model is forced to use 20% from the first step.

## Input

Model:

```text
/mnt/workspace/Qwen3-0.6B
```

Training data:

```text
/mnt/workspace/dclm/**/*.txt
```

The data loader must stream from the full DCLM tree and shard files across DDP
ranks. It must not train repeatedly on one fixed example file.

## Algorithm

For each training batch:

1. tokenize streamed DCLM text into contiguous sequences;
2. run official Qwen3 with patched eager attention;
3. in each attention layer, compute KV-head gate logits from layer input
   hidden states;
4. compute the current keep target from the curriculum:
   - start from `initial_keep_ratio=0.50`;
   - anneal to `target_keep_ratio=0.20`;
   - default anneal length is `30000` steps;
5. apply the default `global_budget` hard gate:
   - sink tokens keep all KV heads;
   - each non-sink token keeps at least its highest-logit KV head;
   - extra token-head slots are filled by global top logits until the layer
     batch reaches `current_keep_ratio`;
6. protect recent query-key pairs with all heads for `gate_recent_tokens_all_heads`
   positions, default `256`;
7. multiply softmax attention probabilities by the straight-through hard gate
   and renormalize, instead of zeroing K/V before attention;
8. compute next-token CE;
9. compute gate budget, load-balance, and z losses;
10. update gate and, by default, official model parameters.

## Metrics

Training logs:

- `train/ce_loss`
- `train/loss`
- `gate/budget_loss`
- `gate/load_loss`
- `gate/z_loss`
- `gate/prob_keep_ratio`
- `gate/hard_keep_ratio`
- `gate/hard_heads_per_token`
- `gate/attention_keep_ratio`
- `gate/current_keep_ratio`
- `gate/head_load_min`
- `gate/head_load_max`
- `gate/head_load_mean`
- `train/tokens_per_second`

## Pass Criteria

The first 20-hour run passes as a training-system test if:

```text
current_keep_ratio follows the 0.50 -> 0.20 curriculum
hard_keep_ratio follows current_keep_ratio
CE remains finite and trends downward or stabilizes
head_load_min is not near zero for most of the run
checkpoints are saved
streaming_data_meta.json reports many DCLM files
```

`hard_keep_ratio` estimates long-term routed KV-cache storage. `attention_keep_ratio`
can be higher because it includes the dense recent-window visibility used during
training.

## Insufficient Evidence

The run is insufficient if:

```text
only a few DCLM files were read
the run ends before gate statistics stabilize
no held-out PPL is measured after training
```

## Next Evaluation

After a checkpoint exists, evaluate:

1. held-out non-DCLM PPL against official Qwen3-0.6B;
2. mean keep ratio on long contexts;
3. downstream multiple-choice accuracy;
4. expected KV-cache memory at 8k, 16k, and 32k.
