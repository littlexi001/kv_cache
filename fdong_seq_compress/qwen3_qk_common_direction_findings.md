# Qwen3 QK Common Direction Findings

Date: 2026-05-29

## 0. Question

前面 K-cache graph probe 发现：

> K-cache 中存在很强 common direction；raw K-K cosine 很高，centered 后相似性明显下降。

这引出一个关键问题：

> 如果 K 之间因为 common direction 都很相似，为什么 qK attention score 仍然有选择性？

本实验验证三个假设：

```text
Exp1: Q 是否和 K common direction 对齐？
Exp2: qK score 的 token 间差异是否来自 centered / residual K？
Exp3: 用 centered K 计算 attention 是否等价于 raw K？
```

## 1. Experiment Setup

输出目录：

```text
fdong_seq_compress/outputs/qk_common_direction_probe_1000
```

配置：

```text
model: fdong/Qwen3-0.6B
text: fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt
sequence length: 1000 tokens
layers: 0-27
q heads: 0-15
kv heads: 0-7
query stride: 8
sampled queries: 125
top-k overlap: 10
```

对每个 layer / Q head，令：

```text
c_t = mean(K_past)
r_i = k_i - c_t
```

比较：

```text
raw_score_i      = q_t · k_i
centered_score_i = q_t · (k_i - c_t)
```

## 2. Main Result

最重要结论：

> K 的 common direction 很大，但它对 qK attention 的 token 间选择性几乎没有贡献。它主要表现为一个对所有历史 token 相同的 additive bias，会在 softmax 中抵消。

数学上：

```text
k_i = c + r_i
q · k_i = q · c + q · r_i
```

其中 `q · c` 对同一个 query 的所有历史 token 都是同一个常数，因此：

```text
softmax(q · k_i) = softmax(q · r_i)
```

这解释了为什么：

```text
raw K-K cosine can be very high
while qK attention remains selective
```

## 3. Exp1: Q 与 K Common Direction 的对齐

K 确实强烈对齐自己的 common direction：

| metric | value |
| --- | ---: |
| mean cos(K, mean K) | 0.791 |
| p50 cos(K, mean K) | 0.810 |
| p95 cos(K, mean K) | 0.986 |

但 Q 和 K common direction 的对齐明显弱得多：

| metric | value |
| --- | ---: |
| mean cos(q, mean K) | 0.108 |
| p50 cos(q, mean K) | 0.062 |
| p95 cos(q, mean K) | 0.340 |
| mean abs cos(q, mean K) | 0.122 |
| p50 abs cos(q, mean K) | 0.067 |

Interpretation:

- Q 通常并不强烈朝向 K common direction。
- 这支持“Q 主要读取 residual / discriminative K subspace”的直觉。
- 但更根本的解释不是 Q 必须正交，而是 common term 在 softmax 中是公共偏置。

## 4. Exp2: qK Score Decomposition

Raw score 与 centered score 的 token 间区分度完全一致：

| metric | value |
| --- | ---: |
| raw score std mean | 2.3572 |
| centered score std mean | 2.3572 |
| centered / raw std ratio | 1.0000 |
| raw-centered score correlation | 1.0000 |
| top-10 overlap | 1.0000 |

Shift error 只有数值误差量级：

```text
raw_score - centered_score - q·c
mean max error: ~4.26e-15
```

Interpretation:

- Centering K 不改变 qK score 的排序。
- Centering K 不改变 qK score 的方差。
- qK score 的选择性来自 residual K，而不是 common K。

## 5. Exp3: Centered K Attention 与 Raw K Attention 等价

直接比较：

```text
softmax(q @ K)
softmax(q @ (K - mean(K)))
```

结果：

| metric | value |
| --- | ---: |
| JS divergence mean | ~0 |
| JS divergence max | ~1.55e-17 |
| top-10 score overlap | 1.0000 |

Interpretation:

> 对同一 query 而言，减去 K 的 mean vector 只是在所有 score 上减去同一个常数，因此 attention distribution 完全不变。

这是一个关键 sanity check：它说明我们用 centered K 来分析 K graph，不是在破坏 attention ranking，而是在去掉一个 attention 本来就会忽略的公共偏置。

## 6. Implication For K-cache Graph

这个结果强烈支持：

```text
K graph should be built in centered / residual K space.
```

不应使用 raw K-K similarity 直接建图，因为 raw similarity 主要受到 common direction 影响；而这个 common direction 对 qK attention 的选择性没有贡献。

更合理的图构建对象是：

```text
r_i = k_i - mean(K_past or K_prefix)
```

然后在 residual K space 中研究：

```text
similarity distribution
distance distribution
in-degree / hubness
attention recall
```

## 7. Current Interpretation

这个现象可以总结为：

> K-cache raw geometry is highly anisotropic, but attention selection operates on the residual geometry after the common component is quotiented out by softmax invariance.

中文表达：

> K 的 raw 空间看起来大家都挤在一个 cone 里，但 qK attention 真正使用的是去掉公共方向之后的 residual 差异。公共方向像一个所有 token 共享的基准电位，softmax 自动把它抵消了。

这让前面的 graph 方向更清楚：

```text
raw K graph: misleadingly dense
centered K graph: closer to attention-relevant address geometry
```

下一步关键实验仍然是：

```text
Can centered K graph candidates recall full qK attention mass?
```
