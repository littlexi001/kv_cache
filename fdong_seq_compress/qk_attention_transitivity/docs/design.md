# QK Attention Transitivity: Design

## Objective

检验真实 Transformer 的单个 attention head 中，QK 高分关系是否具有可用于离散 bucket 的局部传递性。

需要区分三个对象：

1. 参数矩阵是否允许 QK 成为近似内积核；
2. 实际 causal attention 图是否具有两跳闭包；
3. 相似 query 是否检索相似的历史 token 集合。

## Falsifiable Conjecture

对 Qwen3-0.6B 的部分 layer/head，严格历史 attention top-k 图满足：

```text
i attends j
later token l attends i
=> l attends j with probability substantially above a position-matched baseline
```

如果该 lift 跨文本、层和 head 稳定存在，则 QK relation 具有可用于 bucket routing 的局部闭包。若 closure 只等于位置匹配基线，分块更可能来自局部位置结构，而不是 feature-level transitivity。

## Physical Priors

### Prior 1: Similar token states should induce similar QK behavior

如果两个 token state 在一个 head 使用的 feature coordinates 中接近，那么它们产生的 query/key 和 retrieval behavior 也应接近。

### Prior 2: A bucket requires local closure

若 QK 高分关系完全不闭包，把它离散成 cluster 会漏掉大量两跳相关 token。只有局部闭包明显高于基线时，bucket 才有几何基础。

### Prior 3: Raw K common center is not the feature

若 `k_i = c + r_i`，raw K-K 内积中的 `||c||^2` 会让所有 token 看起来相似；但对固定 query，`q^T c` 在 attention softmax 中作为公共偏置抵消。因此参数与几何诊断必须同时报告 raw 和 centered 结果。

## Mathematical Model

令 token state 为行向量 `x_i`。对一个 query head `h` 和它对应的 KV head `g`：

```text
q_i = x_i W_Q,h^T
k_j = x_j W_K,g^T
s_ij = q_i k_j^T
     = x_i M_h x_j^T
M_h = W_Q,h^T W_K,g
```

### Sufficient condition for approximate transitivity

若：

```text
M_h = A_h^T A_h
```

即 `M_h` 对称正半定，则：

```text
s_ij = <A_h x_i^T, A_h x_j^T>
```

QK score 退化为共同 feature space 中的内积。对归一化后的 `z_i = A_h x_i^T / ||A_h x_i^T||`，若：

```text
z_i^T z_j >= 1 - epsilon_1
z_l^T z_i >= 1 - epsilon_2
```

由三角不等式：

```text
z_l^T z_j >= 1 - epsilon_1 - epsilon_2 - 2 sqrt(epsilon_1 epsilon_2)
```

当两个阈值相同为 `epsilon` 时，下界为 `1 - 4 epsilon`。

### Approximate kernel

若：

```text
M_h = P_h + E_h
P_h is symmetric positive semidefinite
```

则误差满足：

```text
|x_i E_h x_j^T| <= ||E_h||_op ||x_i|| ||x_j||
```

因此近似传递性还依赖：

1. 非对称/非 PSD 残差 `E_h` 足够小；
2. token state norm 不剧烈变化；
3. 高分阈值足够严格。

### RoPE complication

实际 attention 使用位置旋转：

```text
s_ij = x_i W_Q,h^T R_i^T R_j W_K,g x_j^T
```

等效 kernel：

```text
M_h(delta) = W_Q,h^T R_delta W_K,g
```

它随相对位置 `delta = j - i` 改变。因此全局传递性需要一个很强的条件：相关距离范围内的 `M_h(delta)` 都近似同一个 PSD kernel。更现实的结果可能是只在固定距离范围或某些 block 内出现局部闭包。

## Implementation Contract

输入：

- 本地 `Qwen3-0.6B`；
- 一段长度 1024 的真实英文文本；
- 代表层 `0, 13, 27`；
- 所有 query heads；
- strict-history top-k，不允许 self token 制造平凡闭包。

输出：

1. 每层每头的权重核诊断；
2. 两跳 attention closure 及随机基线；
3. 相似 query 的 retrieval-set overlap；
4. attention score heatmap；
5. CSV、JSON 和 Markdown 汇总。

## Claim Boundary

本实验只检验一个模型、一段文本上的局部闭包。即使结果为正，也不能直接证明自然语言 feature 是等价类，或 Attention 与 MoE 必须共享同一 bucket。它只回答共享离散 bucket 是否具有 QK 几何基础。
