# K-cache Graph Index Round1 Findings

Date: 2026-05-29

## 0. 文档定位

这个文档合并 Round1 阶段关于 Qwen3 KV cache geometry / K-cache graph index 的主要理解。它取代以下分散文档：

```text
qwen3_kv_cache_geometry_findings.md
qwen3_k_similarity_graph_probe_findings.md
qwen3_k_graph_metric_sweep_findings.md
qwen3_qk_common_direction_findings.md
k_cache_graph_index_experiment_plan.md
```

Round1 的核心问题是：

> Qwen3 的 K-cache 是否具有足够的几何结构，可以进一步研究 graph / anchor / cluster index，从而减少 query 与全量历史 K 的 qK score 计算？

Round1 还没有证明 graph candidate 能召回真实 attention mass。它只回答更前置的问题：

```text
K-cache 是否值得建图？
raw K similarity 是否可信？
centered / residual K 是否更接近 attention-relevant geometry？
K-cache 的相似边是否只是局部平滑，还是存在非局部连接？
```

## 1. 总体结论

Round1 的结论是偏正面的，但边界必须说清楚：

> Qwen3-0.6B 的 K-cache 具备 graph-friendly geometry，尤其是在 centered / residual K space 和部分 layer/head 中；但 query-time graph retrieval 的 attention recall 还未验证。

更具体地说：

- 支持：K-cache 不是无结构点云，centered 后仍有明显 nearest-neighbor structure。
- 支持：K 相似边不全是相邻 token，部分 head 有稳定 long-range edges。
- 支持：部分 layer/head 出现 hub / anchor 候选。
- 支持：K common direction 对 qK attention ranking 不贡献选择性，centered K 更接近 attention-relevant residual geometry。
- 谨慎：raw K cosine / raw dot product 都可能产生 misleading dense graph 或 norm-driven hub。
- 未证明：graph anchors / neighborhoods 能否召回真实 qK attention mass。

因此当前最稳表述是：

```text
K-cache has graph-friendly residual geometry.
Q-to-graph retrieval usefulness remains unverified.
```

## 2. KV 随 Prefix 增长的几何理解

第一轮 prefix-growth geometry 实验把 K/V 的角色区分清楚了：

| 性质 | K-cache | V-cache |
| --- | --- | --- |
| 有效维度 | 随 sequence length sublinear 增长，远慢于 token 数增长 | 也随 sequence length sublinear 增长，但有效维度整体比 K 更高 |
| 各向异性 | 强，存在明显 common direction / cone effect | 相比 K 弱得多，整体更分散 |
| 局部平滑性 | 强，相邻 token 的 K 向量高度相似，更像一条平滑高维轨迹 | 弱，相邻 token 的 V 向量差异明显更大 |
| 小 block 结构 | 支持小尺度连续 block，尤其是 4/8/16 token block | 不支持简单连续 block average |
| 适合承担的角色 | address / index / routing space | content / evidence / information payload |

这直接导向一个结构假设：

```text
K side: build compressed / sparse / graph index
V side: preserve exact or residual content, gather after candidate selection
```

也就是：

```text
K-indexed, V-faithful KV cache compression
```

## 3. Raw K Similarity 为什么不能直接用

实验显示 raw K cosine 很高，主要来自 common direction / cone effect。如果直接基于 raw K 建图，会得到一个过密、区分度弱、可能假 hub 很多的图。

关键解释是：

```text
k_i = c + r_i
q · k_i = q · c + q · r_i
```

对同一个 query 而言，`q · c` 对所有历史 token 是同一个 additive bias，会在 softmax 中抵消。因此 qK attention 的 token 间选择性来自：

```text
q · r_i
```

而不是 common K direction。

Round1 的 QK common direction probe 直接验证了：

- raw score 和 centered score 的 token 间方差一致；
- raw-centered score correlation 为 1；
- top-10 overlap 为 1；
- `softmax(q @ K)` 与 `softmax(q @ (K - mean(K)))` 等价到数值误差。

因此后续 K graph 的默认对象应当是：

```text
centered / residual K
```

而不是 raw K。

## 4. K Similarity Graph 的初步证据

### 4.1 Similarity distribution

在 centered token-level cosine 上，top-k neighbor similarity 仍然明显为正，并且 top-k rank 增大时相似度有合理衰减。

这说明：

> K residual space 中存在可区分的 nearest-neighbor structure，不是所有边都同等强。

它支持 complete weighted K graph 可以删边，保留 sparse stronger edges。

### 4.2 Distance distribution

Centered K top-k edges 不全是局部边。部分设置下，`distance >= 128` 和 `distance >= 256` 的边有稳定比例，尤其在 head-level analysis 中更明显。

这说明：

> K similarity 不只是相邻 token 的局部平滑，也存在非局部连接。

这是 graph memory / nonlocal index 方向的重要正信号。如果相似边全在局部，那这个方向会退化成 local block compression。

### 4.3 In-degree / hubness

Centered K graph 中存在中等 hubness。部分 layer/head，尤其 head-level analysis，出现较明显 hub / anchor 候选。

这支持继续研究：

```text
hub token as graph entry
high in-degree node as anchor
medoid / cluster center
landmark routing
```

但 hubness 本身不是最终证据。理想 anchor 还必须满足：

- incoming edges 不全是局部重复；
- 不是 raw common direction 或 norm artifact；
- 能帮助 query retrieve attention-relevant candidates。

## 5. Metric 层面的 Round1 判断

Round1 比较过 cosine 和 dot product。

当前判断：

```text
primary: centered cosine
diagnostic only: centered dot
next: L2 / PC-removed / whitened K
```

原因：

- raw cosine 被 common direction 污染；
- dot product 混合方向和 norm，容易制造 norm-driven hub；
- centered cosine 去掉 common bias 后更接近 residual address geometry；
- 但 centered cosine 只衡量方向相近，还不能直接保证 qK score stability。

这正是 Round2 引入 L2 distance 的原因。

## 6. Round1 到 Round2 的关键缺口

Round1 支持：

```text
K-K residual graph has promising geometry.
```

但最终目标需要证明：

```text
Q-to-K graph retrieval can cover full qK attention mass.
```

中间还缺两步：

1. 用更直接服务 qK score stability 的 metric 验证 K graph，例如 L2 distance。
2. 构建 graph candidates，与 full qK attention ranking / attention mass 做 recall 对比。

因此 Round2 的核心问题是：

```text
L2 K graph 是否也具有 rank decay、long-range edges、reasonable hubness？
```

如果成立，下一步进入真正的 attention recall。

## 7. 当前推荐路线

当前路线应保持从几何到行为的顺序：

```text
1. centered / L2 / PC-removed / whitened K graph geometry
2. 找 graph-friendly layers / heads
3. 选择 anchors / medoids / cluster centers
4. query scores anchors or graph entries
5. expand selected neighborhoods
6. exact attention over selected K/V
7. measure attention mass recall / top-token recall / CE delta
```

不要过早做复杂 hierarchical tree。先验证一层 graph 是否能召回 attention。如果一层 graph 无效，多层 tree 只是复杂化；如果一层 graph 有效，再做 hierarchy 是自然扩展。

## 8. Round1 保留下来的开放问题

- centered cosine graph 和 L2 graph 是否一致？
- long-range K edges 是否在更长 sequence 上仍然稳定？
- graph-friendly heads 是否稳定跨文本、跨任务存在？
- high in-degree anchors 是否有可解释文本角色？
- graph candidate 是否能 beat local window / random same-size baseline？
- K graph 是否能服务真实 qK attention，而不只是 K-K geometry？

