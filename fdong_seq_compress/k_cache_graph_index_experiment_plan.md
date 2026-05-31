# K-cache Graph Index Experiment Plan

Date: 2026-05-29

## 0. Final Goal

最终目标是：

> 用知识图谱 / 稀疏图索引的结构组织 K-cache，让新 query 不必和所有历史 K 做全量 score，从而降低 K-cache 的计算与访问成本。

目标结构不是把 KV 都压成一个语义摘要，而是：

```text
K side: graph / index / routing structure
V side: selected token or block 的高保真读取
```

也就是：

```text
historical K-cache
-> centered / debiased K graph
-> query retrieves candidate tokens / clusters
-> exact attention over selected K/V
```

因此，当前所有离线实验都服务于一个问题：

> K-cache 的几何结构是否足以支持一个稀疏、可路由、非平凡的图索引？

## 1. Current Research Position

前一轮实验已经给出一个重要结论：

> Raw K similarity 不能直接用于建图，必须去 common direction。

原因是 raw K cosine 过高，明显受到 common direction / cone effect 影响。如果直接用 raw cosine 建图，会得到一个过密、区分度弱、可能是假 hub 很多的图。

所以后续默认设置应为：

```text
centered K
causal top-k
cosine similarity
self excluded
```

后续还应继续比较：

```text
centered K
remove top-PC K
whitened K
```

## 2. Three Core Metrics

### Metric 1: Similarity Distribution

定义：

```text
for each token i:
  find top-k previous tokens j < i by cosine(centered k_i, centered k_j)
  record similarity(i, j)
```

图表：

```text
x-axis: similarity bucket
y-axis: edge count / edge fraction
```

它回答：

> K-cache 里是否存在可用的 nearest-neighbor structure？

适合建图的信号：

- centered top-k similarity 明显高于 0；
- top-k similarity 明显高于 random baseline；
- top-5 / top-10 分布有区分度，而不是所有 pair 都差不多；
- rank1 到 rankk 有合理下降，说明 graph 有局部半径，而不是全空间一团糊。

不适合建图的信号：

- centered top-k similarity 接近 0；
- top-k similarity 和随机高维向量 baseline 差不多；
- 所有 K 之间都非常相似，top-1/top-20 差别很小；
- raw 高、centered 后消失，说明主要是 common direction artifact。

Interpretation:

如果相似性都很高且差不多，单看 K 很难建一个有选择性的图。图会变成“谁都和谁相关”，query routing 没法明显减少候选集。

### Metric 2: Similar-token Distance Distribution

定义：

```text
for each top-k edge i -> j:
  distance = i - j
```

因为是 causal graph，所以 `j < i`，distance 最小为 1。

推荐 bucket：

```text
1
2
3-4
5-8
9-16
17-32
33-64
65-128
129-256
257-512
513+
```

图表：

```text
x-axis: token distance bucket
y-axis: top-k edge count / edge fraction
```

它回答导师最关心的“空间关系”：

> K-space 相似 token 在 sequence axis 上是局部邻居，还是存在远距离连接？

适合知识图谱式建图的信号：

- centered top-k edge 中有稳定比例的 long-range edges；
- `distance >= 128` 或 `distance >= 256` 的边不是极少数；
- 远距离边的 similarity 仍然较高；
- 某些层/头有更强 long-range fraction。

不适合知识图谱式建图的信号：

- 大多数边集中在 `1-8` 或 `1-16`；
- p90/p95 distance 仍然很小；
- long-range edges 很少且 similarity 低。

Interpretation:

如果相似边主要是近邻，那么 K similarity 反映的是局部平滑，适合研究 local block / trajectory compression；如果有明显远距离边，才更支持 graph memory / nonlocal index。

### Metric 3: In-degree Distribution

构造 causal top-k graph：

```text
i -> j if j < i and j is one of i's top-k K neighbors
```

每个 token 的 out-degree 大致由 `top-k` 固定，因此更有信息的是 in-degree：

```text
in_degree[j] = number of future tokens i that choose j as top-k neighbor
```

图表：

```text
x-axis: in-degree bucket
y-axis: number of tokens
```

它回答：

> K graph 是否自然形成 hub / anchor / memory landmark？

适合建图的信号：

- in-degree 分布长尾；
- 少数 token 被大量未来 token 选中；
- top 1% / top 5% nodes 覆盖显著比例的 edges；
- hub 不是纯 raw common direction 造成；
- hub 的 incoming edges 不全是局部邻居；
- hub token 在文本上有可解释性，例如结构边界、重复实体、主题 anchor、特殊符号、任务约束。

不适合建图的信号：

- in-degree 近似均匀；
- 大量 hub 只在 raw K 中出现，centered 后消失；
- hub 只来自非常近距离重复；
- hub 的 incoming similarity 低或不稳定。

Interpretation:

有长尾 hub 不自动证明可以压缩，但它提示可以进一步研究：

```text
hub token as graph entry
hub-centered cluster
cluster representative
landmark routing
```

## 3. How These Metrics Serve The Final Goal

三种指标形成一个筛选链：

```text
similarity distribution
  -> K-space 是否有可区分邻域

distance distribution
  -> 邻域是否只是局部平滑，还是有非局部连接

in-degree distribution
  -> 图是否自然形成 hub / anchor / cluster entry
```

如果三者同时成立：

```text
centered top-k similarity high enough
long-range edge fraction nontrivial
in-degree distribution long-tailed
```

那么可以更有信心进入下一阶段：

```text
build K graph
select candidate nodes/clusters
test q-K attention mass recall
```

如果三者不成立：

```text
similarity weak
edges mostly local
in-degree uniform
```

那知识图谱式 K-cache 可能不是正确方向，更可能转向：

```text
local block compression
sliding / sink / recent-window policy
layer-specific block summaries
```

## 4. The Attention Selectivity Question

一个看似矛盾的问题是：

> 如果不去中心化时 K 之间那么相似，为什么 qK attention score 仍然具有选择性？

可能原因有四个。

第一，K-K cosine 高不等于 q-K score 相同。Attention 用的是：

```text
q · k_i
```

而 K-K cosine 衡量的是：

```text
cos(k_i, k_j)
```

即使 `k_i` 和 `k_j` 都有共同方向，只要它们在 residual direction 上有差异，某些 q 仍然可以放大这些差异。

第二，common direction 对 softmax 可能近似是公共 bias。若：

```text
k_i = c + r_i
```

则：

```text
q · k_i = q · c + q · r_i
```

其中 `q · c` 对所有 token 近似相同，会在 softmax 中被抵消；真正决定 ranking 的可能是 `q · r_i`。

第三，attention score 是 dot product，不是 cosine。K 的 norm、Q 的 direction、RMSNorm/RoPE 等都会影响 score。K-K raw cosine 高只说明方向锥很窄，不说明最终 q-K ranking 没有差异。

第四，模型可能已经学会在 Q 侧读取 residual / discriminative subspace。也就是说，Q 可能天然对 common direction 不敏感，而对 K residual subspace 敏感。

因此，对建图而言：

```text
raw K similarity high
```

不是好消息，反而说明：

```text
graph edge score should be based on debiased K residual geometry
```

这也是为什么后续默认必须 centering，并进一步测试 PC removal / whitening。

## 5. Next Experiments

当前最优先：

```text
centered top-5 similarity distribution
centered top-5 distance distribution
centered top-5 in-degree distribution
```

然后：

```text
top-10 / top-20 sensitivity
head-level version
random baseline
PC removal / whitening
attention mass recall
```

最终关键评估仍然是：

```text
Can K graph candidates cover most q-K attention mass with far fewer tokens?
```

只有这个成立，K-cache graph 才真正服务于降低 attention 计算成本。
