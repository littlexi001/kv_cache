# Round4 Gradient Dynamics

## 0. 目的

本文件回答：

> 高频数据为什么在 Zipf 数据中学得更好？

核心假设：

> Zipf 训练 batch 的真实更新方向更接近高频数据的梯度方向，因此高频 feature 在训练过程中长期主导模型更新。

## 1. 实验设定

在每个 checkpoint 上固定模型参数，构造三类梯度：

```text
g_mix  = grad(loss(B_mix))
g_head = grad(loss_head(B_uniform))
g_tail = grad(loss_tail(B_uniform))
```

其中：

- `B_mix` 按该 run 的 train distribution 采样；
- `B_uniform` 用 uniform eval distribution；
- `loss_head/loss_tail` 只取对应 bucket token 的 loss。

核心指标：

```text
alignment_gap = cos(g_mix, g_head) - cos(g_mix, g_tail)
```

解释：

- `alignment_gap > 0`：真实训练更新方向更接近 head gradient；
- `alignment_gap < 0`：真实训练更新方向更接近 tail gradient；
- 如果 Zipf 条件为正而 uniform 条件不为正，说明这个现象由频率分布触发。

主要结果文件：

- `fdong/experiments/frequency-width-dense-five-analysis.json`

## 2. 关键结果

| run | alignment gap mean | gap range | final cos(g_mix,g_head) | final cos(g_mix,g_tail) | final gap |
|---|---:|---:|---:|---:|---:|
| uniform h64 | -0.2739 | [-0.5980, -0.0684] | 0.3214 | 0.5942 | -0.2728 |
| uniform h96 | -0.2774 | [-0.4701, -0.0289] | 0.2371 | 0.5685 | -0.3315 |
| zipf h64 | 0.1953 | [0.0631, 0.3638] | 0.0492 | -0.1147 | 0.1639 |
| zipf h96 | 0.1397 | [0.0370, 0.2323] | -0.0731 | -0.2145 | 0.1414 |

现象：

1. Zipf 条件下，alignment gap 在所有 checkpoint 都为正。
2. Uniform 条件下，alignment gap 为负。
3. h96 的 Zipf gap mean `0.1397` 小于 h64 的 `0.1953`。

## 3. 结论

结论 1：

> 高频数据效果更好，训练动力学上的直接原因是 Zipf batch 的真实梯度方向长期更接近 head gradient。

支持证据：

- Zipf h64 / h96 的 alignment gap 全程为正；
- uniform h64 / h96 的 alignment gap 为负，说明该指标不是天然偏向 head；
- 因此 head-dominance 来自 Zipf 采样分布本身。

结论 2：

> 加宽可以减弱高频梯度主导，但不能完全消除。

支持证据：

- Zipf h64 gap mean: `0.1953`
- Zipf h96 gap mean: `0.1397`

这说明 h96 的真实更新方向相对更少被 head 独占。

## 4. 与 output margin 的关系

训练动力学上的 head-dominance 最终反映到 output margin 上。

| run | head margin | middle margin | tail margin |
|---|---:|---:|---:|
| uniform h64 | 5.6721 | 5.6560 | 5.6814 |
| uniform h96 | 6.4033 | 6.4634 | 6.4708 |
| zipf h64 | 5.5623 | 4.5416 | 4.0865 |
| zipf h96 | 6.3751 | 5.4523 | 5.0403 |

现象：

- Uniform 条件下 head/tail margin 基本一致。
- Zipf h64 中 tail margin 明显低于 head。
- h96 对 tail margin 的提升大于对 head margin 的提升。

结论：

> 低频 feature 的问题不只是 accuracy 低，而是正确 token 在输出层的 logit margin 更弱；加宽能明显修复 tail margin。

## 5. 本文件结论

Round4 的训练动力学结论是：

> Zipf 分布使真实训练梯度长期偏向高频 feature；这个梯度主导导致低频 feature 的 loss 和 output margin 落后。更宽模型减弱这种梯度主导，并提高 tail margin。
