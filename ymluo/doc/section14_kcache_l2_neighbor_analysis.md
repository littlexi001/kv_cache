# Section 14. K-cache L2 最近邻分析

## 1. 实验目的

前面的 cosine 实验主要回答了一个问题：对于某个 token 的 K 向量，哪些历史 token 的 K 向量在方向上最相似。本节新开一个更保守的 query-agnostic 分析：直接用 L2 距离衡量 K 向量之间的接近程度。

对于每一层、每个 KV head，将 K cache reshape 为：

```text
K_head: [tokens, head_dim]
```

然后对任意两个 token 的 K 向量计算：

```text
d(i, j) = ||k_i - k_j||_2
```

对每个当前 token `i`，在 `j < i` 的历史 token 中选择 L2 距离最小的 top-5 个 token，并画出这些 nearest neighbors 和当前 token 的序列距离：

```text
index_distance(i, j) = i - j
```

这个分析不使用 query，因此它不能直接保证 attention score 不变；但它是“不知道 q 的情况下，哪些 K 向量在所有 bounded query 下最难区分”的保守近似。原因是：

```text
|q · (k_i - k_j)| <= ||q||_2 ||k_i - k_j||_2
```

因此 L2 距离小意味着任意 query 方向上的 dot-product 差异都有较小上界。

## 2. 实验设置

本实验对应新项目：

```text
ymluo/projects/qwen3_kcache_l2_neighbor_analysis
```

核心设置：

```text
model: Qwen3-0.6B
tokens: 5000
neighbor_count: 5
neighbor_scope: previous
variants: raw, centered
rope_max_position_embeddings: 20000
```

运行方式：

```bash
NEIGHBOR_SCOPE=previous \
VARIANTS=raw,centered \
ROPE_MAX_POSITION_EMBEDDINGS=20000 \
bash ymluo/projects/qwen3_kcache_l2_neighbor_analysis/scripts/run_analysis.sh
```

需要注意的是，`ROPE_MAX_POSITION_EMBEDDINGS=20000` 只是确保模型配置允许使用大于 5000 的位置编号；它不会改变 RoPE 在 0-5000 范围内的频率结构，也不会让 5000 以内的位置互相正交。

## 3. 图的含义

典型输出图：

```text
plots/k/centered/layer_16/head_00/index_distance_by_rank_tokens.png
```

横轴：

```text
当前 token index i
```

纵轴：

```text
index_distance = i - j
```

其中 `j` 是在历史 token 中按 L2 距离选出来的 nearest neighbor。因为本实验使用 `neighbor_scope=previous`，所以只允许 `j < i`，图中每个点都表示当前 token 和某个历史最近邻之间的序列距离。

颜色表示 L2 最近邻 rank：

```text
top1: L2 距离最近的历史 K 向量
top2: 第二近
...
top5: 第五近
```

因此，图上的横带 `y≈1000` 不表示所有 token 都接近固定的 token 1000，而表示很多 token 的 nearest neighbor 位于大约 1000 个 token 之前：

```text
j ≈ i - 1000
```

如果所有 token 都接近固定的 token 1000，则图上应出现斜线：

```text
y = i - 1000
```

而不是水平横带。

## 4. raw 与 centered 的关系

本实验同时支持 raw 和 centered：

```text
k_i_centered = k_i - mean(K_head)
```

但是对于 L2 距离，减去同一个中心不会改变两点之间的距离：

```text
||(k_i - mean) - (k_j - mean)||_2
= ||k_i - k_j||_2
```

所以 raw 和 centered 的 L2 nearest-neighbor 结果理论上应该完全一致，最多只有非常小的浮点差异。centered 对 cosine 会产生影响，但对 L2 距离不会改变 nearest neighbor 结构。

因此，如果 centered 图里仍然出现横带，这不是异常，也不说明 center 没有减掉；它说明横带是 K 向量两两 L2 距离本身的结构。

## 5. 浅层现象：固定 lag 横带

在浅层，例如 `L0 H0`，可以观察到明显的水平横带，常见在：

```text
index_distance ≈ 1000
index_distance ≈ 2200
```

这表示很多 token 的 L2 最近邻并不是局部相邻 token，而是稳定地落在某些固定相对距离的位置。由于使用的是 previous scope，这些横带通常在当前 token index 足够大之后才出现。例如 `y≈1000` 的横带需要 `i >= 1000` 才可能出现，因为更早的 token 没有 1000 个历史 token 可选。

这种现象说明浅层 K 向量中存在较强的相对位置几何结构：某些固定 lag 的 K 向量在 L2 空间里更容易接近。

## 6. 深层现象：横带减弱，近邻更局部

用户给出的深层结果包括：

```text
K centered L12 H0
K centered L16 H0
K centered L24 H0
```

对应实验图如下。三张图均来自 `neighbor_scope=previous`、`variants=centered` 的 K-cache L2 最近邻分析；横轴为当前 token index，纵轴为最近邻历史 token 的序列距离 `i - j`，颜色表示 L2 最近邻 rank。

![K centered L12 H0 nearest-L2 index distance](../projects/qwen3_kcache_l2_neighbor_analysis/outputs/k_l2_neighbors/plots/k/centered/layer_12/head_00/index_distance_by_rank_tokens.png)

图 14-1：`K centered L12 H0` 的 nearest-L2 index distance。相比浅层结果，连续横带已经明显减弱，但仍能看到较多离散的远距离 nearest-neighbor 点。

![K centered L16 H0 nearest-L2 index distance](../projects/qwen3_kcache_l2_neighbor_analysis/outputs/k_l2_neighbors/plots/k/centered/layer_16/head_00/index_distance_by_rank_tokens.png)

图 14-2：`K centered L16 H0` 的 nearest-L2 index distance。绝大多数点集中在局部距离附近，远距离点更稀疏，不再形成稳定的 `y≈1000` 或 `y≈2200` 横带。

![K centered L24 H0 nearest-L2 index distance](../projects/qwen3_kcache_l2_neighbor_analysis/outputs/k_l2_neighbors/plots/k/centered/layer_24/head_00/index_distance_by_rank_tokens.png)

图 14-3：`K centered L24 H0` 的 nearest-L2 index distance。深层后段仍有少量长距离 nearest-neighbor，但主要结构已经以局部 nearest-neighbor 为主。

这些图和浅层 `L0 H0` 相比有明显差异：

1. 大部分点集中在 `index_distance≈0` 附近。
2. `y≈1000`、`y≈2200` 这类连续横带明显减弱或基本消失。
3. 仍存在少量远距离散点，尤其在序列后半段，但它们更加稀疏，不形成稳定水平带。
4. L12 的远距离点比 L16/L24 更密集一些；L16 和 L24 更明显地呈现“绝大多数 nearest neighbors 是局部的，少数 token 有长距离 nearest neighbor”的模式。

这说明随着层数加深，K 向量的 L2 最近邻结构从浅层的规则相对位置模式，逐渐转向更内容化或更局部化的表示。换句话说，浅层 K 表示更容易保留 RoPE/位置几何带来的固定 lag 结构，而深层 K 表示中，这种位置周期结构被后续 Transformer 层的内容混合和非线性变换削弱。

## 7. 关于 RoPE 与横带的解释

RoPE 不只有一个单一周期。它在不同维度上使用一组不同频率：

```text
cos(pos * freq_m), sin(pos * freq_m)
```

两个位置之间的相对相位由 `pos_i - pos_j` 决定。当某些相对距离使得该 head 关注的频率维度部分重新接近时，K 向量之间可能在 cosine 或 L2 上变得更接近。

因此，横带可以解释为：

```text
该 head 的 K 表示对某些固定相对距离存在 nearest-neighbor 偏好。
```

但不能简单说“RoPE 周期就是 1000”。更准确的说法是：在该 head 的投影子空间中，RoPE 多频率叠加和模型学习到的 K 投影共同造成了 lag≈1000、lag≈2200 等位置的相似结构。

`ROPE_MAX_POSITION_EMBEDDINGS=20000` 不会消除这个现象，因为它只是扩展可用位置范围，不改变 RoPE 在 0-5000 内的频率几何。要改变横带位置，需要改变 `rope_theta` 或 RoPE scaling 机制，而不是只增大 `max_position_embeddings`。

## 8. 当前结论

根据目前的图，可以得到以下初步结论：

1. L2 最近邻分析确认了 K cache 中存在 query-agnostic 的向量接近结构。
2. 浅层 K 向量存在明显固定 lag 横带，说明浅层 K 的几何结构强烈受到相对位置模式影响。
3. 深层 K 向量中，固定 lag 横带明显减弱，大多数 nearest neighbors 更接近局部 token。
4. raw 和 centered 对 L2 nearest-neighbor 结果不应有实质差异；centered 图仍有横带是符合数学预期的。
5. 深层仍有少量远距离 nearest-neighbor 散点，说明某些 token 仍可能在 K 空间中与远处历史 token 接近，但这种现象不再表现为大面积稳定横带。

## 9. 后续建议

为了进一步确认横带来源，可以继续做以下实验：

1. 画 `index_distance` 的直方图，检查 1000、2200 等 lag 是否形成尖峰。
2. 对不同 layer/head 汇总 top1/top5 的平均 index distance，比较浅层和深层差异。
3. 用随机 token 或打乱文本 token 重跑。如果横带仍存在，说明位置/RoPE 主导；如果横带减弱，说明文本内容结构也有贡献。
4. 改变 `rope_theta` 后重跑。如果横带位置随 `rope_theta` 改变而移动，则可更强地证明 RoPE 频率结构是主要来源。
5. 将 L2 最近邻图和 cosine 最近邻图对齐比较，区分“方向相似”和“包含 norm 的向量距离相似”两种结构。
