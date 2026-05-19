# 第二轮：Attention Readout 与 MoE Selectivity

## 假设

第一轮说明，标准 MoE 不会自然形成可靠的 feature bucket。因此第二轮把问题拆成两个更具体的问题：

1. 当前 attention 是否已经包含有用的 feature retrieval structure？
2. 修改后的 MoE routing 是否能更 selectively 地读出这个结构？

核心想法是：

```text
如果 attention 已经能找到 feature-related tokens，
那么 MoE 不应当从零发现 feature；
它应该读出 attention-derived feature structure。
```

## 实验目的

第二轮测试 attention 是否是有效 feature signal，以及更合适的 router input / router granularity 是否能提升 MoE selectivity。

核心指标包括：

- feature-masked attention 下的 next-token accuracy；
- attention mass 在 local slot 和 higher-level unit 上的占比；
- expert assignment 与 local slot / higher-level unit 的对齐；
- same-expert attention mass；
- 参数矩阵 / 表征空间的奇异方向；
- feature 信息是否能从 attention output 中线性读出。

## 2.1 Feature-Masked Inference

第一个问题是：same-feature KV 是否足以支持预测。

测试的 mask 包括：

| attention mask | loss | accuracy | visible KV |
|---|---:|---:|---:|
| full attention | 0.261 | 91.40% | 100% |
| same local slot occurrence | 2.022 | 74.43% | 9.46% |
| same local slot pattern | 1.978 | 75.08% | 11.25% |
| same higher-level unit | 0.264 | 91.17% | 26.36% |
| random same-size KV | 3.808 | 52.93% | 11.25% |

结论：

```text
lowest-level slot 不是充分 retrieval bucket；
higher-level unit 几乎可以替代 full attention。
```

按位置拆开看可以解释原因：

| mask | local-slot internal accuracy | local-slot boundary accuracy |
|---|---:|---:|
| full attention | 99.58% | 67.85% |
| same local slot | 91.86% | 23.40% |
| same higher-level unit | 99.38% | 67.55% |

Same local slot 在 slot 内部有效，但在 slot boundary 上失败。Same higher-level unit 同时保留了内部预测和 boundary 预测能力。这说明模型依赖的是 higher-level feature retrieval，而不只是 local slot retrieval。

## 2.2 Attention Mass 与 Boundary Noise

Full attention 并没有把所有 mass 都放到 higher-level unit 上：

| layer | local slot history mass | higher-level history mass | higher-level baseline |
|---|---:|---:|---:|
| 0 | 18.47% | 33.94% | 24.45% |
| 1 | 17.53% | 47.63% | 24.45% |
| 2 | 17.74% | 47.07% | 24.45% |

Higher-level unit 只占约 26% 的 visible KV，但在后几层承载约 47% 的 historical attention mass。它没有 100% 集中，但显著高于 baseline，并且足以支持推理。

后续 high-level boundary 分析进一步修正了我们的理解：

| mask | high-level internal accuracy | high-level boundary accuracy |
|---|---:|---:|
| full attention | 96.95% | 8.86% |
| same higher-level unit | 96.74% | 8.69% |
| random same-size KV | 67.14% | 4.49% |

模型在 high-level unit 内部预测准确，但主要在 high-level unit boundary 处失败。这些 boundary 在 synthetic grammar 中是随机 transition，因此本来就没有稳定规律可学。

这改变了我们对“irrelevant attention mass”的理解：

```text
25% attention 落在 ground-truth feature 之外
不一定说明 attention 失败；
它可能是高维噪声，或者模型在尝试解释随机 high-level transition。
```

导师的直觉是：如果高维随机噪声在各维度上分布相似，那么向量点积时这些噪声可以被平均或抵消。因此，分散的 attention mass 可能是无害噪声，而不一定是 MoE 失败的根本原因。

## 2.3 MoE 变体

我们比较了四种 routing 结构：

```text
standard MoE
attention-output router
head-level MoE
attention-output router + head-level MoE
```

其中最有效的是：

```text
attention output w/o residual routing + head-level MoE
```

它提高了 expert assignment 与 attention-relevant token 的一致性：

```text
uniform data: include-self same-expert attention mass ≈ 62.5%
Zipf data:    include-self same-expert attention mass ≈ 59.5%
history-only same-expert attention mass ≈ 40%
```

但它仍然不是可靠的 expert bucket：

```text
MoE-local slot alignment: roughly 35% ~ 43%
MoE-higher-level unit alignment: roughly 28% ~ 34%
```

结论：

```text
MoE 可以部分读出 attention-derived features，
但当前 routing 还不够 selective，不能作为 inverse KV index。
```

## 2.4 Attention Output 上的 Linear Probe

为了区分“attention 缺少 feature”和“MoE 没能读出 feature”，我们在模型表征上训练 linear probe。

Probe 目标：

- local slot id；
- higher-level unit id。

Probe 输入：

- layer input；
- layer output；
- attention output without residual；
- flattened per-head attention output；
- single-head attention output。

Majority baseline 很低：

| target | majority baseline |
|---|---:|
| local slot | 3.8% |
| higher-level unit | 2.4% |

Attention output 中的 feature 显著线性可读：

| layer | representation | local slot probe acc | higher-unit probe acc |
|---|---|---:|---:|
| 0 | attention output w/o residual | 97.7% | 62.8% |
| 1 | attention output w/o residual | 92.5% | 72.7% |
| 2 | attention output w/o residual | 94.1% | 75.8% |
| 0 | flattened head attention output | 98.6% | 66.8% |
| 1 | flattened head attention output | 95.1% | 75.8% |
| 2 | flattened head attention output | 96.6% | 79.1% |

单个 head 能读出一部分 higher-level feature，但没有一个 head 干净地独占完整 higher-level unit。信息分散在多个 head / layer 中。

结论：

```text
attention output 已经包含很强的 ground-truth feature signal；
MoE non-selectivity 不能简单归因于 attention feature 缺失。
```

## 2.5 奇异方向与 Zipf

对 embedding、hidden representation、attention 矩阵和 MoE 矩阵做 raw SVD 后，观察到很强的 common / mean direction：

- embedding raw top singular direction 接近 embedding mean；
- representation raw top direction 经常接近 representation mean；
- final representation top direction 稳定地接近 embedding mean。

去掉 mean 之后：

- local slot 在 hidden representation 中更明显；
- Zipf frequency 主要影响 local / low-level reusable features；
- 高频 local slot 更容易对齐 `k_proj`、MoE `gate_proj` 和 MoE `up_proj` 方向；
- higher-level compositional feature 仍然没有稳定占据 top singular directions。

这说明：

```text
local / frequency feature
-> vector-space feature
-> 体现在 representation norm 和参数方向中

higher-level / compositional feature
-> relational feature
-> 主要体现在 attention score / attention output 中
```

## 第二轮结论

第二轮修正了项目方向：

```text
Attention 不是简单失败了。
Attention output 已经包含有用 feature signal。
当前 MoE 失败更可能是 gate / expert selectivity 问题。
```

因此，在设计新的 attention 约束之前，下一步应先测试：在已有正确 feature signal 的情况下，expert routing 能否通过更合适的训练动力学变得 selective。
