# Experiment Design

## Purpose

Run a first 8-GPU training test for a randomly initialized Qwen3-style routed
top4-head model.

The experiment answers:

```text
Can a model trained from step 0 with per-token top4 head routing reduce CE loss
without router collapse?
```

It does not answer:

```text
Can this architecture match a fully trained Qwen3 model?
Can it reduce real KV cache memory with ragged storage?
```

## Setup

Server paths:

```text
model/tokenizer: /mnt/workspace/Qwen3-0.6B
train text:      /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00000.txt
output root:     /mnt/workspace/routed_top4_qwen3_0p6b_runs
```

Default run command:

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
bash scripts/nohup_train_8x80g.sh
```

TensorBoard:

```bash
bash scripts/tensorboard.sh
```

## Metrics

Primary:

- `train/ce_loss`: next-token cross entropy.
- `train/loss`: CE plus router auxiliary losses.

Router health:

- `router/load_loss`: probability-level balance loss.
- `router/z_loss`: logit scale penalty.
- `router/entropy`: mean gate probability entropy.
- `router/hard_load_min`: minimum selected-token fraction over heads.
- `router/hard_load_max`: maximum selected-token fraction over heads.
- `router/hard_load_mean`: should be close to `top_k / num_heads = 0.25`.

Runtime:

- `train/tokens_per_second`.
- checkpoint save interval and resume status.

## Expected Healthy Pattern

At initialization:

```text
CE should be close to log(vocab_size), about log(151936) = 11.93.
router_hard_load_mean should be close to 0.25.
router entropy should be near log(16) = 2.77 if gate probabilities are uniform.
```

During training:

```text
CE should decrease.
router_hard_load_min should not go near 0.
router_hard_load_max should not approach 1.
router_z_loss should not grow without bound.
```

## First Debug Decisions

If CE becomes NaN:

1. reduce learning rate from `3e-4` to `1e-4`;
2. check whether attention mask has an all-masked query;
3. disable router noise with `--router_noise_std 0.0`.

If router collapses:

1. increase `--router_aux_loss_coef` from `0.01` to `0.05`;
2. increase `--router_noise_std` from `0.1` to `0.2`;
3. inspect `router/hard_load_min` and `router/hard_load_max`.

If training is too slow:

1. reduce `--seq_len` to `1024`;
2. reduce `--gradient_accumulation_steps`;
3. keep checkpointing on, because disabling it may raise memory use sharply.

## Checkpoints

Checkpoints are saved every `500` optimizer steps and once at time-limit stop.

Each checkpoint contains:

```text
model.pt
optimizer.pt
routed_qwen_config.json
trainer_state.json
```

Resume command:

```bash
bash scripts/nohup_train_8x80g.sh \
  --resume_from /mnt/workspace/routed_top4_qwen3_0p6b_runs/<run>/latest_checkpoint
```

## First Result Interpretation

Pass:

```text
CE decreases and router load remains spread over most heads.
```

Partial pass:

```text
CE decreases but router collapses. The architecture can train, but the router
objective is too weak.
```

Fail:

```text
CE does not decrease or training produces NaNs. The operationalization is not
yet stable enough to compare with dense attention.
```
