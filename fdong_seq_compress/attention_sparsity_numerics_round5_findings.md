# Attention Sparsity Numerical Principles: Round5 Findings

Date: 2026-06-11

## 0. 本轮回答的问题

本轮针对 `common_doc/top1_context_research_questions.md` 中“二、为什么”的 A 部分，回答四个数值原理问题：

1. QK score 的分布是平滑的，还是具有长尾、明显间隔或少量极值？
2. score top/tail 1% 分别占据多少 softmax mass？
3. score top/tail token 是否更多位于 common 主奇异子空间或 residual 子空间？
4. score top/tail token 的 V 是对齐、重复、抵消，还是形成不同的输出方向？

当前结论：

| 问题 | 结论 |
|---|---|
| QK score 是否长尾、有间隔或极值？ | 整体分布不算强长尾，但存在少数显著高分极值，且深层更明显。 |
| Score top/tail 覆盖多少 softmax mass？ | score top 1% 平均覆盖 83.7% mass，而 score tail 1% 的 mass 几乎为零。 |
| Score top/tail 是否位于不同奇异子空间？ | 差异很弱，SVD 主子空间不能充分解释 score selectivity。 |
| Score top/tail 的 V 如何影响输出？ | score top 1% 基本恢复 full output；score tail 的方向不同且更易抵消，但实际 mass 极小。 |

## 1. 实验设置

输出目录：

```text
fdong_seq_compress/outputs/attention_sparsity_numerics_20260611_203316
```

配置：

```text
model: Qwen3-0.6B
device: MPS, float16
data: long_english_12000_words.txt
sequence length: 4096
layers: 0, 7, 14, 21, 27
attention heads: all 16 heads
queries: 17 positions sampled from the final 512 tokens
score ratios: 0.1%, 0.5%, 1%, 2%, 4%, 6%, 10%, 20%, 50%
SVD ranks: 1, 4, 8, 16
```

本轮每个 query/head 都重新计算 causal QK score、full softmax probability 和 weighted V output，然后分别分析 score top/tail token。

## 2. QK Score 分布

### 2.1 结果

全局统计：

| 指标 | 平均值 | 中位数 |
|---|---:|---:|
| score skewness | 0.09 | 0.13 |
| excess kurtosis | 0.28 | 0.16 |
| 最高 score 的 z-score | 4.89 | 4.82 |
| 第一名与第二名 score gap / std | 0.97 | 0.95 |

`skewness` 和 `excess kurtosis` 整体不高，因此不能把完整 QK score 分布描述成极端强长尾。但最高 score 通常处于均值以上约 `4-7` 个标准差，说明少量显著极值确实存在。

分层趋势：

| Layer | 最高 score z-score | Top1/Top2 gap / std | Attention effective tokens |
|---:|---:|---:|---:|
| 0 | 3.87 | 0.67 | 163.6 |
| 7 | 3.98 | 0.44 | 96.5 |
| 14 | 4.54 | 0.57 | 54.4 |
| 21 | 5.21 | 1.45 | 29.3 |
| 27 | 6.86 | 1.69 | 10.2 |

### 2.2 结论

> 浅层 score 相对平滑，较多 token 共同参与；随着层数加深，少数高分 winner 逐渐与主体分布拉开，attention 的有效 token 数明显下降。

## 3. Softmax Mass 集中度

### 3.1 结果

| Score 集合 | 平均 softmax mass | 中位数 softmax mass |
|---|---:|---:|
| top 0.1% | 69.8% | 77.0% |
| top 0.5% | 79.7% | 86.7% |
| top 1% | 83.7% | 90.0% |
| top 2% | 87.6% | 92.8% |
| top 4% | 91.1% | 95.2% |
| top 10% | 95.3% | 97.9% |

Score tail 的 mass：

| Score 集合 | 平均 softmax mass |
|---|---:|
| tail 1% | 0.000265% |
| tail 4% | 0.00228% |
| tail 10% | 0.0111% |

这里的 `tail 1%` 指 score 最低的 1%，不是排除 top 1% 后剩余的 99%。

分层上，score top 1% 的平均 mass 从 Layer 0 的 `76.9%` 上升到 Layer 27 的 `93.7%`。

![Softmax mass curve](/Users/bytedance/kv_cache/fdong_seq_compress/outputs/attention_sparsity_numerics_20260611_203316/softmax_mass_curve.png)

### 3.2 结论

> QK score 已经产生少量高分极值，softmax 再把这种差异放大成高度集中的概率质量；深层的选择性强于浅层。

## 4. SVD 子空间

### 4.1 Raw K

Score top/tail 1% 在 raw K 主子空间中的投影能量几乎相同：

| Rank | Top 1% | Tail 1% |
|---:|---:|---:|
| 1 | 70.1% | 70.7% |
| 4 | 79.5% | 80.2% |
| 8 | 85.1% | 85.6% |
| 16 | 90.0% | 90.8% |

这主要反映 raw K 的 strong common direction，而不是 score top token 更位于 common 子空间。

### 4.2 Centered K/V

去掉均值后：

| Representation | Rank | Top 1% | Tail 1% |
|---|---:|---:|---:|
| Centered K | 4 | 48.4% | 47.2% |
| Centered V | 1 | 10.1% | 6.7% |
| Centered V | 4 | 24.3% | 22.1% |
| Centered V | 16 | 49.2% | 47.9% |

V 上有“score top 稍微更靠近主子空间”的弱趋势，但差异较小，且分层上不完全一致。

![SVD projection](/Users/bytedance/kv_cache/fdong_seq_compress/outputs/attention_sparsity_numerics_20260611_203316/svd_projection_top_tail.png)

### 4.3 结论

> Score selectivity 不能被一个全局 SVD 主子空间充分解释。它更可能来自 query-dependent 的 Q-K 方向匹配，而不是所有 score top token 共享一个固定的低维子空间。

## 5. V Output 方向与抵消

### 5.1 Score Top

将 score top 集合内部重新 softmax 后：

| 集合 | 与 full output cosine | Norm ratio |
|---|---:|---:|
| top 0.1% | 0.881 | 1.017 |
| top 1% | 0.972 | 1.009 |
| top 4% | 0.991 | 1.006 |

Score top 1% 已经基本恢复 full attention output 的方向和 norm；score top 4% 更接近 full output。

### 5.2 Score Tail

| 指标 | Top 1% | Tail 1% |
|---|---:|---:|
| 与 full output cosine | 0.972 | 0.184 |
| Top 与 tail output cosine | \- | 0.124 |
| directional cancellation ratio | 0.653 | 0.471 |

`directional cancellation ratio` 越低，表示集合内部 weighted V 越容易相互抵消。因此 score tail 的 V 方向比 score top 更分散。

但是，score tail 1% 在原始 full softmax 中的平均 mass 只有约 `2.65e-6`。因此当前更直接的解释是：

```text
score tail V 的方向不同且更分散，
但其实际贡献首先被 softmax mass 数值性消除。
```

![V direction curve](/Users/bytedance/kv_cache/fdong_seq_compress/outputs/attention_sparsity_numerics_20260611_203316/value_direction_curve.png)

### 5.3 结论

> Score top token 同时拥有高 softmax mass 和较一致的 V 输出方向，因此少量 token 即可恢复 full output；score tail token 的方向不同、内部抵消更强，但其原始 mass 极小。

## 6. 当前数值原理

本轮证据支持如下计算链：

```text
深层 QK score 产生少量显著极值
-> softmax 进一步放大 score 差异
-> score top 1% 获得约 84% mass
-> score top V 聚合方向高度一致
-> score top 1% 基本恢复 full attention output
-> score tail mass 接近零，V 方向更分散且抵消更强
```

因此，当前最有证据支持的 Attention 数值稀疏性来源是：

> **Query-dependent QK 极值 + softmax 放大 + score-top V 的方向一致性。**

## 7. Claim Boundary

当前可以说：

```text
本实验中的 attention output 在 token score 维度上高度稀疏。
深层比浅层具有更强的 score 极值和 softmax mass 集中。
score top 1% 基本恢复 full attention output 的方向和 norm。
全局 SVD 主子空间不能充分解释 score top/tail 的差异。
```

当前还不能说：

```text
所有模型、文本和 query 都具有完全相同的稀疏比例。
score tail token 已经被证明会显著损害 full attention output。
SVD 对任何 query-conditioned 子空间分析都无效。
本轮局部 attention-output 结果已经证明最终任务 loss 的因果机制。
```

本轮只分析一个普通长英文文本、5 个代表层和末尾 17 个 query。跨任务、全层和不同模型规模属于后续适用边界实验。
