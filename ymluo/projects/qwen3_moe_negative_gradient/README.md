# Qwen3 MoE Negative Gradient

This project tests whether explicit inhibition can make MoE routing more
selective on the fdong hierarchical synthetic data.

The intervention adds:

- gate inhibition: current winning router column is treated as the pseudo-label;
  gradient descent pushes other gate columns away from the current router input;
- optional expert repulsion: selected expert groups are regularized so their
  flattened parameter directions are less aligned.

Run:

```bash
bash ymluo/projects/qwen3_moe_negative_gradient/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_negative_gradient/scripts/nohup_train.sh
```

Useful overrides:

```bash
GATE_INHIBITION_WEIGHT=0.1 \
EXPERT_REPULSION_WEIGHT=0.002 \
SYNTHETIC_SAMPLING_DISTRIBUTION=zipf \
MOE_HEAD_LEVEL=true \
MOE_ROUTER_INPUT=attention_output \
bash ymluo/projects/qwen3_moe_negative_gradient/scripts/run_train.sh
```

Metrics are written to:

```text
ymluo/projects/qwen3_moe_negative_gradient/outputs/train/<run_name>/metrics.jsonl
```

Each eval row reports loss, accuracy, per-layer
`same_higher_same_expert`, `local_slot_history_mass`, and
`higher_level_history_mass`.
