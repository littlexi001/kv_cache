# Experiment Design

## Objective

Train a KV-head gate on top of official Qwen3-0.6B and test whether the gate can
drive average KV-head usage toward `20%` without immediate CE collapse.

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
4. apply the default `global_budget` hard gate:
   - sink tokens keep all KV heads;
   - each non-sink token keeps at least its highest-logit KV head;
   - extra token-head slots are filled by global top logits until the layer
     batch reaches `target_keep_ratio`;
5. mask unselected KV head-token slots;
6. compute next-token CE;
7. compute gate budget, load-balance, and z losses;
8. update gate and, by default, official model parameters.

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
- `gate/head_load_min`
- `gate/head_load_max`
- `gate/head_load_mean`
- `train/tokens_per_second`

## Pass Criteria

The first 20-hour run passes as a training-system test if:

```text
hard_keep_ratio approaches 0.20
CE remains finite and trends downward or stabilizes
head_load_min is not near zero for most of the run
checkpoints are saved
streaming_data_meta.json reports many DCLM files
```

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
