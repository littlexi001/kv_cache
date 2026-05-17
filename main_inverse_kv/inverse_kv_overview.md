# Inverse KV Indexing: Current Conclusion

## 1. 核心判断

**我们希望设计 gating 机制（如 `attention output routing + head-level MoE`）实现 inverse KV。**

核心假设是：

```text
语言数据具有 hierarchical / compositional / Zipf distribution
-> Attention 应该捕捉 feature hierarchical relation
-> MoE 应该按照这些 hierarchical feature routing
-> MoE routing 可以作为 Attention KV cache 的 reverse index
```

因此我们真正要验证的是：

> MoE routing 能否从 Attention output 中读出干净的 feature bucket，并用这个 bucket 反向组织 KV cache？

当前结论是：

1. **MoE Selectivity**: Attention output routing + head-level MoE 显著提高了 MoE 分发与 Attention score 的一致性（约 60%），但 MoE bucket 仍不够干净，无法直接作为可靠的 KV cache reverse index。
2. **MoE Selectivity 不高的原因**：Attention 捕捉的 feature hierarchy 不干净。Attention score 有 75% 的注意力捕捉到 ground truth feature hierarchy，但各 hierarchy 混在所有 layer / head 中；同时，仍有约 25% 的注意力分配到无关的 token。
3. **数据分布的影响**：无论数据分布是否有 Zipf，参数矩阵和表征矩阵都是奇异的，且最大奇异方向与 Embedding 矩阵的 mean direction 高度相似。
4. **TODO**：应当设计更干净的 Attention，然后让 MoE gating 直接按 Attention score 分发。

## 2. Attention 学到的 hierarchy 不干净

**Attention 能捕捉 feature hierarchy，但各 hierarchy 混在所有 layer / head 中；同时，真正有用的 local / high-level slot 并没有吃掉全部 attention mass，仍有约 25% 的注意力分配到 ground-truth feature 之外的 token。**

实验中，只保留 same higher-level unit 的 KV，模型推理效果几乎不下降。这说明 higher-level feature 确实是模型预测所需的有效 retrieval bucket。

但问题是，这个 bucket 并没有在 Attention 中以干净形式出现：

- useful local / high-level slot 只占 attention mass 的一部分；
- 即使 high-level slot 足以支持预测，仍有约 25% 左右 attention mass 分配到 ground-truth feature 之外的 token；
- 不同 layer / head 都在混合地学习类似的 hierarchy 信息；
- 没有自然出现“浅层看 local feature、深层看 high-level feature”或“某些 head 专门负责某种 hierarchy”的分工。

因此，Attention 当前不是没有学到 feature，而是：

```text
feature relation exists,
but it is mixed, redundant, and not cleanly indexed.
```

这直接解释了为什么后面的 MoE routing 也不够干净：如果 Attention output 本身就是混合表征，那么 MoE 再从这个 output 上做 routing，很难得到纯净的 feature bucket。

## 3. MoE 的问题来自 Attention 输出的表征不干净

**Token 表征能很好体现 local feature，但不能干净体现 high-hierarchy feature；高层 feature 主要还是体现在 Attention relation 中。因此，MoE 分发不干净的主要原因不是 MoE 完全无效，而是 MoE 依赖的 Attention output 本身就不是干净的 hierarchy representation。**

我们测试了四种结构：

```text
standard MoE
attention-output router
head-level MoE
attention-output router + head-level MoE
```

其中最有效的是：

```text
attention output w/o residual routing + head-level MoE
```

它能让 MoE routing 与 Attention 高相关 token 的一致性明显提高：

```text
uniform data: include-self same-expert attention mass ≈ 62.5%
Zipf data:    include-self same-expert attention mass ≈ 59.5%
history-only same-expert attention mass ≈ 40%
```

这说明：

> 当 routing signal 更接近 Attention output，且 MoE 按 head 拆开后，MoE 确实更容易对齐 Attention 捕捉到的 feature relation。

但它还没有成为可靠 index：

```text
MoE-local slot alignment: roughly 35% ~ 43%
MoE-higher-level unit alignment: roughly 28% ~ 34%
```

也就是说，MoE 不是完全没学到 feature，而是只学到了一个混合版本的 Attention feature。当前失败点不是“MoE 结构完全错了”，而是：

> MoE 依赖 Attention output 分发；Attention output 本身没有提供干净的 hierarchy representation，所以 MoE routing 也不可能干净。

## 4. 参数矩阵和表征空间支持上述判断

**无论数据分布是否有 Zipf，参数矩阵和表征矩阵的最大奇异方向都大量受 common / mean direction 主导；去掉 mean 后，local / frequency feature 更明显，而 high-level compositional feature 仍然主要体现在 Attention score / token relation 中。**

我们分析了 embedding、hidden representation、Attention 参数矩阵、MoE expert 参数矩阵的奇异方向。

首先，raw SVD 的最大奇异方向大量是 common / mean direction：

- embedding raw top singular direction 几乎就是 embedding mean direction；
- representation raw top direction 经常接近 representation mean；
- final representation top direction 与 embedding mean 稳定相似。

这说明最大奇异方向不能直接解释成语义 feature 或 hierarchy feature。

去掉 mean direction 后，观察到的结构是：

- local slot 在 hidden representation 中很可分；
- Zipf frequency 对 local / low-level reusable feature 有影响；
- 高频 local slot 更容易对齐 `k_proj`、MoE `gate_proj / up_proj` 的重要方向；
- higher-level compositional feature 没有稳定占据参数矩阵 top singular directions；
- higher-level feature 主要体现在 Attention score / token relation 中。

因此更准确的 feature 图景是：

```text
local / frequency feature
-> vector-space feature
-> visible in representation norm, k_proj, MoE gate/up directions

higher-level / compositional feature
-> relational feature
-> mainly visible in attention score / attention pattern
```

这进一步支持我们对失败原因的判断：

> 现在 MoE routing 分不干净，不是因为 hierarchy 不存在，而是因为 high-level hierarchy 没有以干净的单点表征形式暴露给 router。

## 5. Inverse KV 的工程问题

**MoE 本身依赖 Attention 的计算结果，且发生在 Attention 之后，因此当前层的 gating 结果不能直接作为该层 Attention KV cache 的 reverse index。它可能对后续层或下一步结构设计有用，但若要服务当前层 KV selection，必须改变 routing 与 attention 的计算顺序。**

理想的 inverse KV 需要：

```text
first compute / predict routing index
-> select KV candidates
-> perform attention inside selected candidates
```

而当前结构是：

```text
attention
-> attention output
-> MoE routing
```

因此，当前 MoE routing 无法反过来指导已经发生的 attention retrieval。

这并不说明 inverse KV 不可行，而是说明：

> 不能直接把当前 MoE output 当 inverse KV index；需要先让 Attention relation 更干净，并将 routing / index 前置到 Attention 之前。

## 6. TODO

**下一步的核心不是继续调 MoE，而是设计更干净的 Attention，并让 MoE / routing 按 Attention 的结果分发。**

1. 设计新的 Attention 约束，让不同 layer / head 捕捉更干净、更可分离的 hierarchy feature。

2. 避免所有 layer / head 混合学习同一套 hierarchy，尝试形成更明确的分工：

```text
some heads/layers -> local feature
some heads/layers -> higher-level feature
some heads/layers -> boundary / transition feature
```

3. 在 Attention relation 更干净之后，把 Attention score / Attention-derived correlation 显式作为 routing index。

4. 将新的 routing index 前置到 Attention 之前，用它做 KV candidate selection。

5. 验证该结构是否对 inverse KV cache / catastrophic forgetting 有效。

6. 进一步评估 memory saving 和 latency saving。
