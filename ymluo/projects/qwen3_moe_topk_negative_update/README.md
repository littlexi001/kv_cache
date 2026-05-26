# Qwen3 MoE Top-k Negative Update

This project studies a top-k MoE update rule:

```text
router selects K experts per token
the first P selected expert slots receive normal gradients
the remaining K-P selected expert slots receive reversed gradients
```

Default configuration:

```text
MOE_NUM_UNIQUE_EXPERTS=16
MOE_NUM_EXPERTS_PER_TOK=4
NEGATIVE_UPDATE_PRIMARY_SLOTS=1
NEGATIVE_UPDATE_SECONDARIES=true
NEGATIVE_UPDATE_SCALE=1.0
```

So by default each token selects 4 experts from 16. The highest-weight selected
expert is updated normally, and the other 3 selected experts receive reversed
gradients.

Implementation detail: the forward value still uses the selected experts and
their top-k weights. Only the backward gradient through secondary expert outputs
is reversed:

```text
flipped = expert_output.detach() - scale * (expert_output - expert_output.detach())
```

With `scale=1.0`, this is exact gradient ascent for the secondary selected
experts while preserving the same forward activations.

## Data Modes

The dataset can be switched by `SYNTHETIC_DATA_MODE`:

```text
hierarchical          original vertical hierarchical synthetic data
structured_language   topic/entity/noise/copy/bridge synthetic data
```

Default is `structured_language`.

## Run

Structured-language default:

```bash
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

Hierarchical vertical synthetic data:

```bash
SYNTHETIC_DATA_MODE=hierarchical \
SEQ_LEN=128 \
DEBUG_VOCAB_SIZE=257 \
RUN_NAME=topk-negative-hierarchical \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/nohup_train.sh
```

Eval:

```bash
CKPT_FILE=ymluo/projects/qwen3_moe_topk_negative_update/outputs/train/topk-negative-structured/checkpoints/10000.pth \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_eval.sh
```

## Useful Overrides

Change expert count, selected experts, and normally updated slots:

```bash
MOE_NUM_UNIQUE_EXPERTS=32 \
MOE_NUM_EXPERTS_PER_TOK=8 \
NEGATIVE_UPDATE_PRIMARY_SLOTS=2 \
RUN_NAME=topk-negative-32e-top8-primary2 \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

Disable reversed gradients while keeping top-k MoE:

```bash
NEGATIVE_UPDATE_SECONDARIES=false \
RUN_NAME=topk-baseline-16e-top4 \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

Reduce reverse-gradient strength:

```bash
NEGATIVE_UPDATE_SCALE=0.25 \
RUN_NAME=topk-negative-scale025 \
bash ymluo/projects/qwen3_moe_topk_negative_update/scripts/run_train.sh
```

Main metrics are inherited from `moe_selectivity_experiment.py`:

- `eval_loss`, `eval_acc`;
- `same_higher_by_layer`;
- `higher_mass_by_layer`;
- `expert_load_by_layer`;
- `load_balance`.
