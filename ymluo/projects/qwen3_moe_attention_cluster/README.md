# Qwen3 MoE Attention Cluster

This project uses the original fdong hierarchical synthetic data and adds a
router regularizer based on attention neighborhoods.

Core idea:

```text
If token i attends strongly to token j, then token i and token j should have
similar router distributions and high probability of using the same expert.
```

For KV-cache compression experiments, the default runner now uses a
pre-attention router. The router can read `layer_input`, `q`, `k`, or `v`, while
the expert input remains the normal post-attention MLP input. Attention weights
are only used as a training teacher for the auxiliary loss; inference can route
from the pre-attention feature before doing full QK retrieval.

Set `MOE_EXPERT_INPUT_ATTENTION_TOPK > 0` to feed the experts from a sparse
attention output: the model first selects the top-k attended K/V positions per
query/head from the full attention weights, renormalizes those weights, and uses
that V-weighted output for the MLP/expert input. The main attention residual
still uses full attention, so this isolates the expert-input ablation.

The auxiliary loss is differentiable. It does not hard-code a ground-truth
expert id. For each layer, it computes:

```text
same_expert_prob(i, j) = dot(router_prob_i, router_prob_j)
loss = - attention(i, j) * log(same_expert_prob(i, j))
```

To avoid the trivial solution where every token enters one expert, the project
also supports a negative-pair term:

```text
negative pairs = token pairs from different higher-level units
loss_neg = -log(1 - dot(router_prob_i, router_prob_j))
```

This pushes different higher-level units away from each other in router space.
By default the negative feature layer is `1`, which is the higher-level unit id
in the fdong hierarchical metadata.

By default only the top attended history tokens are used:

```text
ATTENTION_CLUSTER_TOPK=4
ATTENTION_CLUSTER_INCLUDE_SELF=false
ATTENTION_CLUSTER_DETACH_ATTENTION=true
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.01
```

Run:

```bash
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/nohup_train.sh
```

The top of the nohup log contains the resolved `run_train.sh` configuration, so
good runs can be traced back to their exact shell hyperparameters.

Useful overrides:

```bash
ATTENTION_CLUSTER_WEIGHT=0.1 \
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0.01 \
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.01 \
ATTENTION_CLUSTER_TOPK=8 \
MOE_HEAD_LEVEL=false \
USE_PRE_ROUTER=true \
PRE_ROUTER_INPUT=q \
MOE_EXPERT_INPUT_ATTENTION_TOPK=8 \
RUN_NAME=attn-cluster-w01-top8 \
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_train.sh
```

Important defaults:

```text
SYNTHETIC_BLOCK_SIZE=4
SYNTHETIC_NUM_HIERARCHY_LAYERS=2
SYNTHETIC_NUM_UNITS_PER_LAYER=64
SYNTHETIC_SAMPLING_DISTRIBUTION=zipf
SYNTHETIC_ZIPF_ALPHA=1.1
MOE_HEAD_LEVEL=false
USE_PRE_ROUTER=true
PRE_ROUTER_INPUT=q
PRE_ROUTER_CONTROLS_ATTENTION=false
MOE_EXPERT_INPUT_ATTENTION_TOPK=0
MOE_NUM_EXPERTS_PER_TOK=1
GATE_INHIBITION_WEIGHT=0.0
ATTENTION_CLUSTER_WEIGHT=0.01
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.01
EXPERT_REPULSION_WEIGHT=0.0
```

Main metrics:

- `same_higher_same_expert`: same higher-level unit routed to the same expert;
- `same_higher_by_layer`: per-layer version of the same metric;
- `higher_mass_by_layer`: attention mass on same higher-level history tokens;
- `expert_load_by_layer`: per-layer expert load;
- `attn_cluster`: training value of the attention-cluster auxiliary loss.
- `attn_neg`: training value of the different-higher-unit negative-pair loss;
- `load_balance`: training value of optional MoE load-balance loss.

Eval a checkpoint:

```bash
CKPT_FILE=ymluo/projects/qwen3_moe_attention_cluster/outputs/train/moe-attention-cluster/checkpoints/10000.pth \
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_eval.sh
```
