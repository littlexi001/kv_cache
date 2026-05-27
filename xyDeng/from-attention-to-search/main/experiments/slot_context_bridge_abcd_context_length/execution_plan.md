# Execution Plan: abcd_context_length_bridge

Anchor:

```text
../../problem_anchors/gated_main_causes/slot_context_dominance_router_specialization_anchor.md
```

## 1. Run Goal

结论先行：这个 bridge 实验只回答一个窄问题：

```text
0524 AB/CB semantic-prior decay 和当前 slot-context NMI 上升是否来自任务/损失定义差异？
```

具体测试：

```text
在 r-B / A-B / C-B / D-B 四个 context roles 下，
把训练流程、B-position 评估、top-1 selected-gate MoE、seeds、steps、batch size、lr
都对齐当前 slot-context 实验，
比较 short context 与 long context 是否增加 B-position route-slot NMI，
并比较 random init 与 semantic/slot-centroid init。
```

这个实验不是为了替代当前结论，也不是证明真实语言 specialization。它是一个 bridge control，用来判断“旧实验 NMI 下降”和“当前实验 NMI 上升/保持”是否真矛盾。

## 2. Data Construction

使用 4 个 context roles 和 4 个 experts：

```text
role 0: r B_i -> Y_{0,i}
role 1: A B_i -> Y_{1,i}
role 2: C B_i -> Y_{2,i}
role 3: D B_i -> Y_{3,i}
```

其中 `B_i` 在所有 roles 中共享，并要求：

$$
Y_{0,i},Y_{1,i},Y_{2,i},Y_{3,i}\ \text{distinct for the same}\ B_i.
$$

因此 `B_i` alone 不能预测 target，context role 必须参与预测。

数据条件：

| Condition | Prefix before `B_i` | Purpose |
|---|---|---|
| `short_role` | one role token near B | 对齐当前 `C0_short` |
| `long_repeated_role` | same role token repeated 5 times | 测试 context length / strength 是否提升 B hidden role signal |
| `long_distributed_role` | 5-token distributed role code | 测试不是只靠一个 repeated token shortcut |

建议固定 motif 位置：

```text
positions 10-14: context prefix
position 15: B_i
position 16: Y_{role,i}
seq_len = 32
```

这样与当前 slot-context 实验完全对齐；旧 0524 的随机 motif 位置不作为本 bridge 的默认设置。

## 3. Model / Training / Evaluation Setup

模型：

```text
top1_selected_gate_sparse
experts = 4
load balance = off
dropout = 0
same d_model / ffn_dim / n_heads as current slot-context run
```

训练：

```text
full-sequence causal NTP
loss = mean CE(x_t -> x_{t+1})
steps = 1600
batch size = 384
lr = 0.0008
seeds = 20260521, 20260522, 20260523, 20260524
```

初始化条件：

| Init | Meaning |
|---|---|
| `random_init` | no router semantic prior |
| `semantic_role_centroid_init` | collect B-position hidden states grouped by context role and initialize router rows by whitened role centroids |

不要加入：

```text
slot-router auxiliary loss
supervised routing CE
load balance regularizer
oracle router in main discussion figure
```

Oracle 可以作为 artifact-only upper bound，但不要放进 meeting 主图。

## 4. Metrics And Artifacts

主判断指标：

```text
per-seed B-position NMI(route, role), step 0 -> final
per-seed route-role heatmap, step 0 vs final
```

辅助指标：

```text
target-position CE / accuracy
full-sequence NTP CE
role probe accuracy from h_B
assignment-utility agreement
forced expert target CE heatmap
ablation delta heatmap
router weight delta from init
role center norm / pairwise cosine trajectory
```

主图只放一张：

```text
random init vs semantic_role_centroid_init
step 0 vs final
all seeds
short_role and long_repeated_role shown separately, no seed averaging, no oracle
```

## 5. Decision Rules

支持“当前结果和 0524 不矛盾，如果 context target pressure 对齐后 NMI 可上升”的模式：

```text
long_repeated_role + semantic init 的 B-position NMI 高于 short_role 和 random init；
target-position accuracy 接近 1；
route-role heatmap 在 fixed/shared-B setting 下更接近 role-to-expert permutation；
utility heatmap 不与 routing 结论冲突。
```

支持“0524 prior decay 更一般，当前 slot result 是特殊构造”的模式：

```text
即使 full NTP + fixed B-position target-pressure 对齐，
semantic init 的 role NMI 仍从 step 0 衰减到 random-level；
long context 不提升 final NMI；
target 已学会但 route-role alignment 不稳定。
```

重定向模式：

```text
NMI 上升但 assignment-utility 不升：context helps routing alignment, not functional specialization.
NMI 不升但 probe 上升：hidden role signal 可见，但 router 不使用。
random init 也升：NTP target pressure alone can discover role routing, semantic init is not necessary in this bridge.
```

## 6. How This Compares To 0524 And Current Runs

旧 0524：

```text
AB/CB setup; final-position classification objective;
B-context roles = B_in_B_only / B_in_AB / B_in_CB;
semantic init creates early routing but final NMI decays.
```

当前 slot-context：

```text
full-sequence causal NTP;
B-position next token is role/slot-dependent;
fixed motif position;
4 slots and 4 experts;
slot-centroid init at B position.
```

bridge:

```text
use old r-B / AB / CB / DB conceptual roles
but align objective and B-position target pressure with current experiment.
```

因此这个 bridge 能区分：

```text
old decay caused by objective/data mismatch
vs
semantic/context init generally fails even under aligned NTP target pressure
```

## 7. Result Location

如果执行，结果写入：

```text
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/summary.md
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/detailed.md
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/figures/
Projects/from-attention-to-search/main/experiments/slot_context_bridge_abcd_context_length/tables/
```

sync closure 可追加到：

```text
Projects/from-attention-to-search/XingyuD/sync/0526_slot_context_dominance/report.md
```
