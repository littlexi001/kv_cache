# Qwen3 MoE Forced Warmup

This project tests an oracle warmup for MoE routing on fdong hierarchical
synthetic data.

For the first `FORCED_WARMUP_STEPS`, each input position `i` is forced to route
to:

```text
floor(i / higher_unit_len) % num_experts
```

With the default synthetic setup, `higher_unit_len = block_size ^
num_hierarchy_layers = 4 ^ 2 = 16`, so all 16 tokens inside the same top-level
unit occurrence are sent to the same expert during warmup.

The implementation also adds a router cross-entropy loss during warmup so the
gate learns the forced assignment. After warmup, routing returns to normal MoE
top-k routing.

Run:

```bash
bash ymluo/projects/qwen3_moe_forced_warmup/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_forced_warmup/scripts/nohup_train.sh
```

Useful overrides:

```bash
FORCED_WARMUP_STEPS=200 \
FORCED_WARMUP_ROUTER_LOSS_WEIGHT=1.0 \
FORCED_WARMUP_HIGHER_UNIT_LEN=16 \
MOE_NUM_UNIQUE_EXPERTS=4 \
bash ymluo/projects/qwen3_moe_forced_warmup/scripts/run_train.sh
```

Resume from an existing checkpoint:

```bash
INIT_CHECKPOINT=ymluo/projects/qwen3_moe_forced_warmup/outputs/train/moe-forced-warmup/checkpoints/1000.pth \
RUN_NAME=forced-warmup-resume \
bash ymluo/projects/qwen3_moe_forced_warmup/scripts/run_train.sh
```

Eval rows in `metrics.jsonl` report loss, accuracy,
`same_higher_same_expert`, `local_slot_history_mass`,
`higher_level_history_mass`, and expert load metrics.
