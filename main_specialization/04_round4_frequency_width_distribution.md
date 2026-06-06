# Round4 Distribution Evidence

## 0. 目的

本文件回答两个问题：

1. 高频/低频效果差异是不是由频率分布本身触发？
2. 更宽模型是否对低频 feature 的改善显著大于高频 feature？

## 1. 实验设定

数据：

- synthetic dataset: `HierarchicalPatternData`
- feature: top hierarchy unit
- train distribution: uniform or Zipf
- eval distribution: uniform
- bucket: 按 train distribution 中 top-unit frequency 分为 `head/middle/tail`

模型：

- dense Transformer
- 2 layers
- hidden size: 64 / 96
- intermediate size: 2x hidden
- steps: 1000

主要结果文件：

- `fdong/experiments/frequency-width-dense-five-analysis.json`
- `fdong/experiments/frequency-width-skew-zipf0p7-bucket-eval-step1000.json`
- `fdong/experiments/frequency-width-skew-zipf1p0-bucket-eval-step1000.json`
- `fdong/experiments/frequency-width-skew-zipf1p3-bucket-eval-step1000.json`
- `fdong/experiments/frequency-width-skew-zipf1p6-bucket-eval-step1000.json`

## 2. 均匀数据 vs Zipf 数据

| run | head loss | middle loss | tail loss | head acc | tail acc |
|---|---:|---:|---:|---:|---:|
| uniform h64 | 0.2812 | 0.2771 | 0.2791 | 0.9087 | 0.9114 |
| uniform h96 | 0.2728 | 0.2690 | 0.2702 | 0.9112 | 0.9132 |
| zipf h64 | 0.2413 | 0.4008 | 0.5111 | 0.9332 | 0.8972 |
| zipf h96 | 0.2253 | 0.3641 | 0.4450 | 0.9355 | 0.9006 |

现象：

- Uniform 条件下，head/middle/tail loss 基本一致。
- Zipf 条件下，head loss 明显低于 tail loss。
- h64 的 Zipf tail-head loss gap 是 `0.2697`。
- h96 的 Zipf tail-head loss gap 是 `0.2197`。

结论：

> 高频/低频效果差异由频率不均匀触发。均匀分布下没有明显 head/tail gap；Zipf 分布下 gap 清楚出现。

## 3. Zipf alpha sweep

固定总训练 token 和模型设置，只改变 Zipf alpha。

| alpha | h64 head loss | h64 tail loss | h64 tail-head gap | h96 head loss | h96 tail loss | h96 tail-head gap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 0.2295 | 0.3421 | 0.1126 | 0.2227 | 0.3263 | 0.1036 |
| 1.0 | 0.2270 | 0.4034 | 0.1764 | 0.2177 | 0.3728 | 0.1550 |
| 1.3 | 0.2413 | 0.5111 | 0.2697 | 0.2253 | 0.4450 | 0.2197 |
| 1.6 | 0.2761 | 0.8258 | 0.5497 | 0.2459 | 0.5859 | 0.3400 |

现象：

- h64 tail-head gap 随 alpha 单调增大：`0.1126 -> 0.1764 -> 0.2697 -> 0.5497`。
- h96 tail-head gap 也增大，但增长更慢。

结论：

> 频率越不均匀，tail 数据效果越差。频率 skew 是这个现象的直接因果变量。

## 4. 宽度对 tail 的收益更大

| condition | h64 -> h96 head loss improvement | h64 -> h96 middle loss improvement | h64 -> h96 tail loss improvement |
|---|---:|---:|---:|
| uniform | 0.0084 | 0.0081 | 0.0089 |
| zipf alpha=0.7 | 0.0068 | 0.0120 | 0.0158 |
| zipf alpha=1.0 | 0.0092 | 0.0183 | 0.0306 |
| zipf alpha=1.3 | 0.0160 | 0.0367 | 0.0661 |
| zipf alpha=1.6 | 0.0302 | 0.0865 | 0.2399 |

现象：

- Uniform 条件下，h96 对 head/middle/tail 的改善几乎一样。
- Zipf 条件下，h96 对 tail 的改善显著大于 head。
- skew 越强，h96 的 tail-side advantage 越大。

结论：

> 大模型/宽模型的优势不是均匀改善所有数据；在 long-tail 分布下，它对低频数据的边际改善显著更大。

## 5. 本文件结论

Round4 当前最强分布证据是：

1. head/tail gap 在 uniform 条件下基本消失，在 Zipf 条件下显著出现；
2. Zipf alpha 越大，tail-head gap 越大；
3. width 对 tail 的改善随 Zipf alpha 增大而显著放大。

因此可以说：

> 高频和低频数据的效果差异是频率导致的；频率越不均匀，tail 越差；模型变宽后，低频数据的改善程度显著大于高频数据。
