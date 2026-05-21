# Qwen3 MoE Attention Cluster

This project uses the original fdong hierarchical synthetic data and adds a
router regularizer based on attention neighborhoods.

Core idea:

```text
If token i attends strongly to token j, then token i and token j should have
similar router distributions and high probability of using the same expert.
```

The auxiliary loss is differentiable. It does not hard-code a ground-truth
expert id. For each layer, it computes:

```text
same_expert_prob(i, j) = dot(router_prob_i, router_prob_j)
loss = - attention(i, j) * log(same_expert_prob(i, j))
```

By default only the top attended history tokens are used:

```text
ATTENTION_CLUSTER_TOPK=4
ATTENTION_CLUSTER_INCLUDE_SELF=false
ATTENTION_CLUSTER_DETACH_ATTENTION=true
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
ATTENTION_CLUSTER_TOPK=8 \
MOE_HEAD_LEVEL=false \
MOE_ROUTER_INPUT=attention_output \
RUN_NAME=attn-cluster-w01-top8 \
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_train.sh
```

Important defaults:

```text
SYNTHETIC_BLOCK_SIZE=4
SYNTHETIC_NUM_HIERARCHY_LAYERS=2
SYNTHETIC_NUM_UNITS_PER_LAYER=64
SYNTHETIC_SAMPLING_DISTRIBUTION=uniform
MOE_HEAD_LEVEL=false
MOE_NUM_EXPERTS_PER_TOK=1
GATE_INHIBITION_WEIGHT=0.0
ATTENTION_CLUSTER_WEIGHT=0.05
EXPERT_REPULSION_WEIGHT=0.0
```

Main metrics:

- `same_higher_same_expert`: same higher-level unit routed to the same expert;
- `same_higher_by_layer`: per-layer version of the same metric;
- `higher_mass_by_layer`: attention mass on same higher-level history tokens;
- `expert_load_by_layer`: per-layer expert load;
- `attn_cluster`: training value of the attention-cluster auxiliary loss.

Eval a checkpoint:

```bash
CKPT_FILE=ymluo/projects/qwen3_moe_attention_cluster/outputs/train/moe-attention-cluster/checkpoints/10000.pth \
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_eval.sh
```
