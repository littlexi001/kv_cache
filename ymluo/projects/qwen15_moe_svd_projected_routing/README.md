# Qwen1.5 MoE SVD Projected Routing

This project trains a resized Qwen1.5-MoE-style model with routing logits built
from projections of each layer input onto the input-side singular vectors of an
attention projection matrix.

Default base model/config source:

```text
/mnt/workspace/Qwen1.5-MoE-A2.7B
```

The default preset shrinks the model to a roughly 0.6B-class MoE before random
initialization and changes the MoE layout to:

```text
1 + n1 + n2 + n3 experts = 1 + 16 + 24 + 8 = 49 experts
active experts per token = 1 + 2 + 3 + 1 = 7 experts
```

## Routing

For each decoder layer, let `x` be the layer input and let the selected
attention projection be `W`. `torch.nn.Linear` stores weights as
`[out_features, in_features]`, so for `y = x @ W.T` the input-side singular
directions are the columns of `V` from:

```text
W = U S V.T
```

The patch computes:

```text
x_f = x @ V
```

and uses four feature ranges:

```text
expert0:       x_f[0]       -> 1x1 gate, always selected
group n1:      x_f[2%-10%]  -> gate to 16 experts, top-2 selected
group n2:      x_f[11%-70%] -> gate to 24 experts, top-3 selected
group n3:      x_f[71%-100%]-> gate to 8 experts, top-1 selected
```

The selected expert outputs all have hidden-size output and are combined by the
router weights. Normal attention still runs for every token.

`--projection_source` can be `q`, `k`, `v`, or `o`; the default is `q`.

## Data And Metrics

The default run uses a lightweight structured synthetic next-token dataset and
logs:

```text
loss_lm
accuracy
loss_load_balance
```

For real text, set `DATA_MODE=text` and point `DATA_PATH` at a directory of
`*.txt` files.

## Run

```bash
bash ymluo/projects/qwen15_moe_svd_projected_routing/scripts/nohup_train.sh
```

Useful overrides:

```bash
PROJECTION_SOURCE=o \
SVD_REFRESH_INTERVAL=500 \
DATA_MODE=synthetic \
MAX_STEPS=1000 \
bash ymluo/projects/qwen15_moe_svd_projected_routing/scripts/nohup_train.sh
```

```bash
DATA_MODE=text \
DATA_PATH=/mnt/workspace/dclm \
bash ymluo/projects/qwen15_moe_svd_projected_routing/scripts/nohup_train.sh
```

Resume:

```bash
RESUME_FROM_CHECKPOINT=auto \
bash ymluo/projects/qwen15_moe_svd_projected_routing/scripts/nohup_train.sh
```
