# Round4 Frequency-Width Conclusion

## 0. Round4 要回答的问题

老板的问题是：

> 为什么大模型 scale up 时通常更倾向于显著加宽，而不是只加深？

Round4 先不直接讨论所有真实大模型能力，而是把问题压到一个可控 synthetic setting：

> 当数据本身有 Zipf / long-tail 频率分布时，模型宽度是否会特别改善低频 feature 的学习？

这里的“大/小”暂时不定义成总参数量，而定义成：

> 表征空间、并行子空间和可分解特征容量的大小。

在 dense Transformer 中，本轮主要操作 hidden size：`h64` vs `h96`。

## 1. 当前最稳结论

Round4 现在支持四个结论。

### 1.1 高频/低频效果差异由频率分布触发

均匀数据下，head/middle/tail 的 loss 基本一致；Zipf 数据下，head 明显好于 tail。

更强的是 alpha sweep：固定总训练 token 和模型设置，只改变 Zipf alpha，tail-head gap 单调增大。

| alpha | h64 tail-head loss gap | h96 tail-head loss gap |
|---:|---:|---:|
| 0.7 | 0.1126 | 0.1036 |
| 1.0 | 0.1764 | 0.1550 |
| 1.3 | 0.2697 | 0.2197 |
| 1.6 | 0.5497 | 0.3400 |

结论：高低频数据效果不同不是随机训练噪声，而是由频率 skew 系统性触发；频率越不均匀，tail 越差。

详细见 [Round4 Distribution Evidence](./04_round4_frequency_width_distribution.md)。

### 1.2 训练动力学机制来自高频数据对梯度方向的主导

定义：

```text
g_mix  = 当前训练分布 batch 的梯度
g_head = uniform batch 中只取 head token loss 的梯度
g_tail = uniform batch 中只取 tail token loss 的梯度

alignment_gap = cos(g_mix, g_head) - cos(g_mix, g_tail)
```

Zipf 条件下，`alignment_gap` 在所有 checkpoint 都为正；uniform 条件下为负。这说明 Zipf 训练的真实更新方向长期更接近高频数据希望的方向。

| run | alignment gap mean | final gap |
|---|---:|---:|
| uniform h64 | -0.2739 | -0.2728 |
| uniform h96 | -0.2774 | -0.3315 |
| zipf h64 | 0.1953 | 0.1639 |
| zipf h96 | 0.1397 | 0.1414 |

结论：高频数据学得更好，核心训练动力学证据是高频数据主导真实训练梯度方向。h96 的 gap 小于 h64，说明加宽会减弱这种 head dominance，但不会完全消除。

详细见 [Round4 Gradient Dynamics](./04_round4_frequency_width_gradient_dynamics.md)。

### 1.3 这种变化不是不可逆的

两个 intervention 都显示 tail failure 很大程度可逆。

**Zipf + inverse-sqrt frequency loss reweight：**

| run | head loss | tail loss | tail-head gap |
|---|---:|---:|---:|
| baseline zipf h64 | 0.2413 | 0.5111 | 0.2697 |
| reweight h64 | 0.2443 | 0.3763 | 0.1320 |
| baseline zipf h96 | 0.2253 | 0.4450 | 0.2197 |
| reweight h96 | 0.2286 | 0.3467 | 0.1181 |

**Zipf -> uniform fine-tune：**

| run | step | head loss | tail loss | tail-head gap |
|---|---:|---:|---:|---:|
| source zipf h64 | 1000 | 0.2413 | 0.5111 | 0.2697 |
| fine-tune h64 | 50 | 0.2904 | 0.3543 | 0.0638 |
| fine-tune h64 | 300 | 0.2731 | 0.2913 | 0.0181 |
| source zipf h96 | 1000 | 0.2253 | 0.4450 | 0.2197 |
| fine-tune h96 | 50 | 0.2811 | 0.3047 | 0.0236 |
| fine-tune h96 | 300 | 0.2662 | 0.2739 | 0.0076 |

结论：Zipf 阶段造成的 tail 落后不是完全不可逆的表征损伤；通过 loss 权重补正或恢复 uniform 数据续训，可以显著缓解甚至接近解决。

详细见 [Round4 Intervention Evidence](./04_round4_frequency_width_interventions.md)。

### 1.4 更宽模型对低频数据的改善显著大于高频数据

在 uniform 数据中，h96 相对 h64 对三类 bucket 的提升几乎一样，loss 改善都约 `0.008`。

在 Zipf 数据中，h96 的提升明显集中在 tail：

| condition | h64 -> h96 head loss improvement | h64 -> h96 middle loss improvement | h64 -> h96 tail loss improvement |
|---|---:|---:|---:|
| uniform | 0.0084 | 0.0081 | 0.0089 |
| zipf alpha=1.3 | 0.0160 | 0.0367 | 0.0661 |
| zipf alpha=1.6 | 0.0302 | 0.0865 | 0.2399 |

结论：宽度的收益不是均匀地提升所有 feature；当频率分布不均匀时，宽度对 tail 的边际收益显著大于 head。这个结果支持“更宽模型更能承接 long-tail 更新、缓解低频学习瓶颈”。

详细见 [Round4 Distribution Evidence](./04_round4_frequency_width_distribution.md) 和 [Round4 Intervention Evidence](./04_round4_frequency_width_interventions.md)。

### 1.5 理论解释：宽度改善 feature-gradient kernel 的几何

两层线性模型 `f(x)=a^T W x` 中，即使函数类本身仍是线性的，宽度 `m` 也会改变参数空间中的梯度几何。对 one-hot feature `i`，有：

```text
<grad f_i, grad f_j> = <w_i, w_j>,  i != j
```

随机初始化下：

```text
cos(grad f_i, grad f_j) = O_p(1 / sqrt(m))
```

因此更宽的参数矩阵会让不同 feature 的梯度方向更接近正交，feature-gradient kernel 更接近对角化。Zipf 数据中 tail feature 的自更新项 `p_t ||grad f_t||^2` 本来最弱，因此最怕高频 feature 的 off-diagonal 交叉干扰；宽度降低这种干扰后，tail 获得最大边际收益。

详细见 [Round4 Linear Theory](./04_round4_frequency_width_linear_theory.md)。

## 2. 机制表述

当前最准确的机制表述是：

> Zipf 分布通过样本曝光和有效梯度权重，使优化过程长期偏向高频 feature，造成低频 feature 的 output margin 和 loss 落后。这个落后很大程度上能被 reweight 或 uniform fine-tune 修复，说明它不是完全不可逆的表征损伤。但宽度仍然重要，因为 skew 越强，h96 对 tail 的额外改善越大，说明更宽模型更能承接长尾更新并缓解频率不均衡带来的学习瓶颈。

换成更短的版本：

> 训练动力学是主因，宽度是缓解长尾梯度/表征瓶颈的结构条件。

## 3. 支持但需要保守表述的证据

SVD/PCA 和 probe 提供了辅助证据，但不是当前最硬证据。

SVD/PCA 显示：

| run | final effective rank | final top10 explained variance |
|---|---:|---:|
| uniform h64 | 47.00 | 0.3741 |
| uniform h96 | 62.06 | 0.3200 |
| zipf h64 | 44.56 | 0.4067 |
| zipf h96 | 60.25 | 0.3326 |

这支持“更宽模型有更高 effective rank，Zipf 会让表征更集中”。但它还不能单独证明“高频主方向污染 tail 表征”。

Probe 显示 h96 的表征整体更可分，但 tail-specific probe 改善不够强，因此不能作为主要机制证据。

## 4. 文件组织

Round4 当前文件：

1. [Round4 Frequency-Width Conclusion](./04_round4_frequency_width_conclusion.md)
2. [Round4 Distribution Evidence](./04_round4_frequency_width_distribution.md)
3. [Round4 Gradient Dynamics](./04_round4_frequency_width_gradient_dynamics.md)
4. [Round4 Intervention Evidence](./04_round4_frequency_width_interventions.md)
5. [Round4 Linear Theory](./04_round4_frequency_width_linear_theory.md)

实验代码入口：

- `fdong/scripts/run_frequency_width_analysis_pipeline.sh`
- `fdong/scripts/run_frequency_width_solution_pipeline.sh`

主要结果文件：

- `fdong/experiments/frequency-width-dense-five-analysis.json`
- `fdong/experiments/frequency-width-reweight-inverse_sqrt-analysis.json`
- `fdong/experiments/frequency-width-zipf-to-uniform-analysis.json`
- `fdong/experiments/frequency-width-skew-zipf{0p7,1p0,1p3,1p6}-bucket-eval-step1000.json`
