# Tail 10 Query Tokens 的 Top 1% Attention Token 分析

## 1. 数据来源

本节分析的文件是：

`C:/Users/夕/Documents/niah_len2000_depth25_tail10_top1_tokens_by_query.csv`

样本为：

`niah_len2000_depth25`

它是 Needle-in-a-Haystack 数据中的一个样本，context 约 2000 words，needle 插入在约 25% 深度处。这里分析的是 prompt 最后的 10 个 query token，在每一层、每个 head 中 attention score 最高的 top 1% key tokens。

该文件共有 4480 行，对应：

- 28 层；
- 每层 16 个 attention heads；
- 每个 head 观察 tail 10 个 query tokens；
- 每一行表示一个 `(layer, head, query token)` 的 top 1% key token 集合。

tail 10 个 query tokens 是：

| token_index | token_text |
| ---: | --- |
| 2715 | ` best` |
| 2716 | ` thing` |
| 2717 | ` to` |
| 2718 | ` do` |
| 2719 | ` in` |
| 2720 | ` San` |
| 2721 | ` Francisco` |
| 2722 | `?\n` |
| 2723 | `Answer` |
| 2724 | `:` |

也就是说，这里主要看的是模型在读到问题末尾：

```text
What is the best thing to do in San Francisco?
Answer:
```

时，每层每个 head 的 top 1% attention key tokens 到底落在什么位置。

## 2. 位置分类定义

每个被选中的 key token 被分到四类之一，分类是互斥的，优先级如下。

| 类别 | 含义 |
| --- | --- |
| `answer_span` | key token 位于前文中的答案序列，即 expected answer 在 context 中出现的位置。 |
| `front_1pct` | key token 位于当前 query 可见 index 的最前 1%。 |
| `tail_1pct` | key token 位于当前 query 可见 index 的最后 1%。 |
| `other` | 不属于以上三类的其他位置。 |

注意：`answer_span` 优先级最高。如果一个 token 既在 answer span 里，又刚好落在前 1% 或尾部 1%，也会被归为 `answer_span`。

## 3. Overall Summary

全局统计如下：

| 类别 | token_count | percentage |
| --- | ---: | ---: |
| `answer_span` | 1990 | 1.59% |
| `front_1pct` | 15052 | 12.00% |
| `tail_1pct` | 45550 | 36.31% |
| `other` | 62848 | 50.10% |
| **Total** | 125440 | 100.00% |

整体上看，tail 10 query tokens 的 top 1% attention 并不是主要落在答案 span 上。最大的一类是 `other`，约 50.10%；其次是 `tail_1pct`，约 36.31%。这说明大部分 head 在这些 query token 上仍然强烈依赖局部尾部信息和分散的中间上下文。

不过，`answer_span` 仍然有 1.59%。考虑到答案 span 在整个 2700 多 token 序列中占比很小，这个比例不能简单理解为低价值；它说明某些层和 head 确实会把答案位置纳入 top 1%。

## 4. 逐层统计

| Layer | Answer span | Front 1% | Tail 1% | Other | 主要现象 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.00% | 0.04% | 40.07% | 59.89% | 分散到中间其他位置 |
| 1 | 0.02% | 2.79% | 44.08% | 53.10% | 混合分布 |
| 2 | 0.00% | 3.84% | 40.92% | 55.25% | 混合分布 |
| 3 | 0.31% | 9.40% | 36.90% | 53.39% | 混合分布 |
| 4 | 0.16% | 4.82% | 50.02% | 45.00% | 强尾部/近邻偏置 |
| 5 | 0.02% | 4.02% | 56.29% | 39.67% | 强尾部/近邻偏置 |
| 6 | 0.74% | 9.46% | 33.06% | 56.74% | 混合分布 |
| 7 | 0.36% | 10.76% | 51.52% | 37.37% | 强尾部/近邻偏置 |
| 8 | 0.80% | 10.69% | 31.87% | 56.63% | 混合分布 |
| 9 | 0.74% | 10.65% | 29.00% | 59.62% | 分散到中间其他位置 |
| 10 | 0.04% | 4.93% | 58.08% | 36.94% | 强尾部/近邻偏置 |
| 11 | 1.03% | 9.38% | 24.58% | 65.02% | 分散到中间其他位置 |
| 12 | 1.23% | 12.14% | 27.83% | 58.79% | 分散到中间其他位置 |
| 13 | 1.54% | 13.26% | 28.86% | 56.34% | 混合分布 |
| 14 | 2.68% | 12.39% | 31.90% | 53.04% | 开始稳定命中答案证据 |
| 15 | 1.61% | 16.76% | 41.83% | 39.80% | 混合分布 |
| 16 | 4.42% | 19.55% | 28.21% | 47.81% | 答案证据命中较明显 |
| 17 | 4.11% | 14.75% | 34.22% | 46.92% | 答案证据命中较明显 |
| 18 | 3.57% | 18.21% | 33.28% | 44.93% | 开始稳定命中答案证据 |
| 19 | 2.81% | 20.96% | 33.15% | 43.08% | 开始稳定命中答案证据 |
| 20 | 3.93% | 16.23% | 21.54% | 58.30% | 开始稳定命中答案证据 |
| 21 | 2.57% | 16.38% | 24.67% | 56.38% | 开始稳定命中答案证据 |
| 22 | 1.29% | 19.71% | 33.97% | 45.02% | 头部/sink 比例较高 |
| 23 | 1.81% | 14.53% | 36.43% | 47.23% | 混合分布 |
| 24 | 2.57% | 18.82% | 30.27% | 48.35% | 开始稳定命中答案证据 |
| 25 | 3.17% | 16.32% | 25.36% | 55.16% | 开始稳定命中答案证据 |
| 26 | 2.90% | 19.35% | 29.06% | 48.68% | 开始稳定命中答案证据 |
| 27 | 0.00% | 5.83% | 59.78% | 34.40% | 强尾部/近邻偏置 |

## 5. 分层解读

### 5.1 Layer 0-2：低层主要看局部和普通上下文

Layer 0-2 几乎不命中答案 span：

- Layer 0: 0.00%
- Layer 1: 0.02%
- Layer 2: 0.00%

这几层的 top 1% token 主要落在 `tail_1pct` 和 `other`。其中 `tail_1pct` 在 40%-44% 左右，说明低层 head 对最近 token、格式 token、问题附近 token 有明显偏好。

这一段可以理解为：低层更多处理局部形式、词面关系和 prompt 末尾结构，还没有稳定承担长距离检索答案证据的功能。

### 5.2 Layer 3-10：尾部偏置仍然明显，答案命中很弱

Layer 3-10 中，答案 span 比例仍然很低，大多低于 1%。同时，若干层表现出很强的尾部偏置：

- Layer 4: `tail_1pct` 50.02%
- Layer 5: `tail_1pct` 56.29%
- Layer 7: `tail_1pct` 51.52%
- Layer 10: `tail_1pct` 58.08%

这说明在中低层，tail query tokens 的 top 1% attention 仍然大量保留问题末尾和 `Answer:` 附近的 token。模型此时还不是主要在回看 needle，而是在强化当前问题和输出位置的局部上下文。

### 5.3 Layer 11-15：开始出现答案证据，但仍不占主导

Layer 11-15 的答案 span 比例开始上升：

- Layer 11: 1.03%
- Layer 12: 1.23%
- Layer 13: 1.54%
- Layer 14: 2.68%
- Layer 15: 1.61%

尤其 Layer 14 是一个明显转折点，答案 span 上升到 2.68%。这说明从中层开始，部分 head 已经会把前文中的答案序列放入 top 1%。

但这阶段答案证据仍然不是主导项。`other` 和 `tail_1pct` 仍然占很大比例，说明模型同时依赖大量非答案位置的信息。

### 5.4 Layer 16-21：答案证据命中最明显的阶段

Layer 16-21 是本样本中最值得关注的区间。

答案 span 比例最高的层集中在这里：

- Layer 16: 4.42%
- Layer 17: 4.11%
- Layer 18: 3.57%
- Layer 19: 2.81%
- Layer 20: 3.93%
- Layer 21: 2.57%

Layer 16 是全层最高，达到 4.42%。这说明在处理 tail question tokens 时，模型最明显地把答案 span 纳入 top 1% 的位置，大约出现在中高层。

这也符合一个直觉：低层关注局部形式，高层逐渐整合语义和检索信息，中高层开始把问题 token 与远处 needle answer 对齐。

### 5.5 Layer 22-26：答案命中仍存在，但注意力更分散

Layer 22-26 中，答案 span 仍然保持在 1.29%-3.17%：

- Layer 22: 1.29%
- Layer 23: 1.81%
- Layer 24: 2.57%
- Layer 25: 3.17%
- Layer 26: 2.90%

这说明答案证据在高层仍然被保留，但不如 Layer 16-20 那么突出。与此同时，`front_1pct` 在多层中较高，例如 Layer 22 和 Layer 26 接近 20%。这可能反映了高层 head 对 prompt 头部或 attention sink 的利用。

### 5.6 Layer 27：最后一层重新强烈偏向尾部

Layer 27 的分布很特殊：

- `answer_span`: 0.00%
- `front_1pct`: 5.83%
- `tail_1pct`: 59.78%
- `other`: 34.40%

最后一层几乎不再把答案 span 放入 top 1%，而是强烈偏向 tail 1%。这可能说明最后一层更多服务于局部输出格式、当前位置预测和最终 logits 形成，而不是直接承担远距离证据检索。

## 6. 主要结论

第一，tail 10 query tokens 的 top 1% attention token 不是纯答案证据。整体上，答案 span 只占 1.59%，而 `tail_1pct` 和 `other` 合计超过 86%。

第二，答案证据命中具有明显层间结构。低层几乎不命中答案；中层开始出现；Layer 16-21 最明显；最后一层又回到强尾部偏置。

第三，top 1% 中包含多类 token：局部问题 token、`Answer:` 附近 token、attention sink / 前部 token、答案 span token、以及大量其他上下文 token。因此，“top 1% 有效”不能解释为“top 1% 全是答案”，而更像是模型保留了一组混合路由，其中少量关键 head 和层负责把答案证据纳入可用集合。

第四，对于这个样本，如果要人工检查模型是否真的检索到答案，应优先看 Layer 16-21，尤其是这些层里 `answer_span_pct` 高的 heads，而不是只看 Layer 0 或最后一层。

## 7. 后续查看建议

建议后续重点打开以下文件：

```text
tail10_position_layer_head_summary.csv
```

按照 `answer_span_pct` 从高到低排序，找出最像“答案检索 head”的层和 head。然后再回到：

```text
tail10_top1_tokens_by_query.csv
```

查看这些 head 对应行里的 `selected_token_texts`，确认它们是否真的选中了类似：

```text
eat a sandwich and sit in Dolores Park
```

这样的答案序列。

