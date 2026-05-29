# Qwen3 K-cache Graph Metric Sweep Findings

Date: 2026-05-29

## 0. Question

本轮实验要回答的问题是：

> 当前模型的 K-cache 是否具有适合建图的性质？更具体地说，能否把 K-cache 从 complete weighted graph 稀疏化成一个带 anchor / cluster / long-range edge 的图结构，从而为后续 query routing 降低 K-cache 计算成本？

最终目标不是“证明 K-K 相似性存在”本身，而是：

```text
centered K graph
-> anchors / cluster centers / sparse neighborhoods
-> query first scores anchors
-> expand selected neighborhoods
-> exact attention over selected K/V
```

本轮实验还没有测试 query-time attention recall，因此它只能判断：

```text
K-cache 是否具备建图的初步几何条件
```

不能直接证明：

```text
K graph 已经可以替代 full q-K attention
```

## 1. Experiment Setup

输出目录：

```text
fdong_seq_compress/outputs/k_graph_metric_sweep_20260529_191716
```

配置：

```text
model: fdong/Qwen3-0.6B
text: fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt
sequence length: 1000 tokens
layers: 0-27
heads: 0-7
centered: true
top-k: 10, 20, 50
similarity: cosine, dot product
analysis level: token-level, head-level
```

每个实验同时输出三类指标：

```text
similarity distribution
token-distance distribution
in-degree distribution
```

## 2. Bottom Line

本轮结果的结论是：

> 当前 K-cache 部分支持我们想做的 graph / anchor / cluster index。它支持继续推进 centered K graph 原型，尤其是 head-wise centered cosine graph；但还没有支持到“可以直接用 anchor tree 替代 full q-K attention”。

更具体地说：

- 支持：centered K 中存在可区分的 top-k neighbor structure。
- 支持：高相似边不全是局部边，存在显著 long-range edges。
- 支持：部分层/头出现 hub / anchor 候选。
- 谨慎：dot product 图可能受到 K norm artifact 影响，不能直接当作最可信图。
- 未证明：graph anchors 能否召回真实 query attention mass。

因此当前最稳的判断是：

```text
K-K graph has promising geometry.
Q-to-graph retrieval usefulness remains unverified.
```

## 3. Evidence 1: Similarity Distribution

Centered token-level cosine 仍然有明显 top-k structure：

| setting | mean | p50 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| token cos top-10 | 0.5397 | 0.5239 | 0.7981 | 0.9987 |
| token cos top-20 | 0.4707 | 0.4481 | 0.7552 | 0.9987 |
| token cos top-50 | 0.3724 | 0.3434 | 0.6817 | 0.9987 |

Interpretation:

- Centering 后相似性没有坍缩到 0，说明 K-space 里存在可用 nearest-neighbor structure。
- top-k 增大时 mean / median 合理下降，说明强边和弱边之间有一定区分度。
- 这支持“complete weighted graph 可以删边留下 stronger edges”。

这一步支持建图的第一条件：

> edge weights are not all equally weak or equally strong after debiasing.

## 4. Evidence 2: Distance Distribution

Centered token-level cosine 的 long-range edge 比例：

| setting | distance p50 | distance p95 | frac >=128 | frac >=256 |
| --- | ---: | ---: | ---: | ---: |
| token cos top-10 | 18 | 397 | 0.249 | 0.087 |
| token cos top-20 | 35 | 462 | 0.307 | 0.120 |
| token cos top-50 | 77 | 561 | 0.402 | 0.178 |

Interpretation:

- K similarity 不是纯局部平滑；`>=128` 的边有稳定比例。
- top-k 越大，long-range edge fraction 越高，说明远距离候选进入更宽的 neighborhood。
- 这支持 K-cache 中存在非局部图连接，而不只是 local block trajectory。

Head-level 更强。Centered head-level cosine top-10 中，long-range fraction 最高的 heads 包括：

| head | frac >=128 | frac >=256 | distance p50 | distance p95 |
| --- | ---: | ---: | ---: | ---: |
| L06H3 | 0.602 | 0.348 | 187 | 660 |
| L15H1 | 0.576 | 0.356 | 163 | 609 |
| L02H6 | 0.569 | 0.320 | 136 | 687 |
| L11H1 | 0.547 | 0.294 | 163 | 611 |
| L03H5 | 0.541 | 0.304 | 157 | 610 |

This is one of the strongest positive signals:

> Some heads naturally form long-range K-neighbor graphs.

这说明后续不应该只看 all-head token-level concat；更合理的是：

```text
select graph-friendly heads
build graph per selected head/layer
test query recall
```

## 5. Evidence 3: In-degree / Anchor Structure

Token-level cosine 的 hubness 中等：

| setting | p95 in-degree | max in-degree | zero frac | top 1% edge frac | top 5% edge frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| token cos top-10 | 21 | 66 | 0.010 | 0.037 | 0.134 |
| token cos top-20 | 42 | 110 | 0.004 | 0.037 | 0.137 |
| token cos top-50 | 108 | 238 | 0.002 | 0.034 | 0.135 |

Interpretation:

- 有 hub / anchor 苗头，但 token-level graph 的 hubness 不极端。
- top 5% nodes 覆盖约 13-14% edges，说明存在一定集中度，但不是少数点统治全图。

Head-level cosine 的 hubness 更明显：

| setting | p95 in-degree | max in-degree | zero frac | top 1% edge frac | top 5% edge frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| head cos top-10 | 22 | 138 | 0.014 | 0.047 | 0.155 |
| head cos top-20 | 46 | 221 | 0.005 | 0.046 | 0.157 |
| head cos top-50 | 116 | 372 | 0.003 | 0.040 | 0.151 |

某些 head 的 hubness 更强。例如 centered head-level cosine top-10：

| head | top 5% edge frac | max in-degree | zero frac |
| --- | ---: | ---: | ---: |
| L02H7 | 0.291 | 138 | 0.057 |
| L02H5 | 0.275 | 123 | 0.035 |
| L02H1 | 0.266 | 109 | 0.021 |
| L02H3 | 0.241 | 100 | 0.017 |
| L02H4 | 0.235 | 107 | 0.029 |

Interpretation:

- Layer 2 的多个 heads 有明显 anchor / hub 候选。
- 这支持用 hub / medoid / high in-degree node 作为 graph entry 或 cluster center 的想法。
- 但 hubness 还需要和 distance 结合看：理想 anchor 不应只是局部重复 token。

## 6. Dot Product Results Are Less Trustworthy

Dot product sweep 也显示出 long-range edges 和 hubness，但更可能受到 norm artifact 影响。

例如 head-level dot top-10 中，有些 heads 的 top 5% nodes 覆盖 70%-80% edges，同时 zero in-degree 很高：

```text
L00H2 top5% edge frac = 0.804, zero frac = 0.773
L00H6 top5% edge frac = 0.797, zero frac = 0.748
L02H7 top5% edge frac = 0.797, zero frac = 0.505
```

这可能意味着少数大-norm K 吸走了大量边，而不是形成了更好的 semantic / routing graph。

因此当前建议：

```text
primary graph score: centered cosine
secondary diagnostic: centered dot
future score: PC-removed / whitened cosine
```

## 7. Does This Support Our Anchor / Tree Intuition?

我们的直觉是：

```text
complete weighted K graph
-> delete weak edges
-> retain sparse high-similarity graph
-> identify anchors / cluster centers
-> query scores anchors first
-> expand selected neighborhoods
-> exact attention over selected K/V
```

本轮实验支持其中前三步：

```text
delete weak edges: supported by centered similarity distribution
retain nonlocal sparse graph: supported by distance distribution
identify anchors: partially supported by in-degree distribution
```

但还没有支持最后两步：

```text
query scores anchors first
expand selected neighborhoods
exact attention recall remains high
```

因此当前结论应表述为：

> K-cache has graph-friendly geometry, especially in selected heads, but query-routing utility is not yet proven.

## 8. Recommended Next Experiment

下一步应该直接验证：

> graph anchors / neighborhoods 能否召回真实 q-K attention mass？

最小实验：

```text
1. Choose graph-friendly heads:
   e.g. high long-range fraction heads such as L06H3, L15H1, L02H6,
   and high hubness heads such as L02H7.

2. Build centered cosine top-k graph from historical K.

3. Choose anchors:
   high in-degree nodes, medoids, or cluster centers.

4. For each query q_t:
   full attention ranking = q_t · K_past
   graph candidate set = anchors + anchor neighborhoods

5. Measure:
   attention mass recall
   top-attended-token recall
   candidate fraction
   recall vs candidate budget
```

Success condition:

```text
small candidate set covers most full attention mass
```

If this succeeds, then the graph is not just geometrically interesting; it becomes a plausible mechanism to reduce K-cache attention cost.

## 9. Current Final Judgment

本轮实验结果支持继续推进 K-cache graph index。

最推荐的 next prototype setting：

```text
centered K
cosine similarity
head-wise graph
top-10 or top-20 graph edges
focus on graph-friendly heads
evaluate attention recall before any architecture change
```

不建议现在就做：

```text
raw K graph
dot-only graph
all-layer all-head uniform graph
直接用 anchors 替代 full attention without recall test
```

最终一句话：

> 当前模型的 K-cache 具备建图的初步好性质；尤其某些 head 中存在非局部长距离边和 hub 候选。但是否能降低计算成本，取决于下一步 q-to-graph attention recall 实验。

## 10. Follow-up: Why Raw K Is Dense But qK Attention Is Selective

后续 QK common direction probe 进一步解释了一个关键现象：

> K 中存在巨大 common direction，导致 raw K-K similarity 很高；但 qK attention score 仍然具有选择性。

实验记录见：

```text
fdong_seq_compress/qwen3_qk_common_direction_findings.md
```

核心结果：

```text
mean cos(K, mean K) = 0.791
p50  cos(K, mean K) = 0.810
p95  cos(K, mean K) = 0.986

mean cos(q, mean K)     = 0.108
p50  cos(q, mean K)     = 0.062
mean abs cos(q, mean K) = 0.122
```

更关键的是，raw qK score 与 centered-K qK score 完全等价：

```text
raw score std mean      = 2.3572
centered score std mean = 2.3572
centered/raw std ratio  = 1.0000
score correlation       = 1.0000
top-10 overlap          = 1.0000
attention JS divergence ~= 0
```

数学解释：

```text
k_i = c + r_i
q · k_i = q · c + q · r_i
```

其中 `q · c` 对同一个 query 的所有历史 token 都是同一个常数，会在 softmax 中抵消：

```text
softmax(q · c + q · r_i) = softmax(q · r_i)
```

因此：

> K 的 common direction 对 raw K-K similarity 影响很大，但对 qK attention 的 token 间选择性几乎没有贡献。

这强化了本轮 graph sweep 的方法论结论：

```text
raw K graph is misleadingly dense;
centered / residual K graph is the attention-relevant address geometry.
```
