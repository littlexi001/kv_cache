# Inverse KV Overview

## 总体目标

我们希望模型形成更好的 expert selectivity：具有相同或相近 feature 的 token 应当被路由到稳定的 expert bucket 中。

如果这个目标成立，MoE routing 就不只是 FFN 计算中的负载分发机制，而可以进一步成为 attention KV cache 的 reverse index：decode 时可以根据当前 token 的 routing bucket，优先检索同 bucket 的历史 KV，从而减少无关 KV 的访问。

这个工作目前的核心链条是：

```text
hierarchical / compositional sequence feature
-> attention 捕捉 feature relation
-> MoE gate / expert 形成 feature bucket
-> expert bucket 作为 KV reverse index
```

当前实验结论是：attention 已经较好捕捉了 hierarchy feature，但 baseline MoE 没有自然形成足够干净的 expert bucket。因此后续重点从“attention 是否有 feature”转向“为什么 gate / expert 没有把 feature 转化成稳定 selectivity”。

## Round 1：标准 Attention / MoE 是否自然捕捉 Feature

文件：[inverse_kv_round1_notes.md](./inverse_kv_round1_notes.md)

结论：

- Attention 对 ground-truth slot feature 有明显敏感性，有 75% 的注意力 mass 集中在 same-slot 中。
- 标准 MoE routing 基本没有自然形成和 local slot / higher-level slot 对齐的 expert bucket。

这一轮确认：标准 Attention 能看到 feature，但标准 MoE 不会自然给出我们想要的 feature-level routing。

## Round 2：Attention 是否已经提供可用 Feature Signal

文件：[inverse_kv_round2_notes.md](./inverse_kv_round2_notes.md)

结论：

- Attention 捕捉到的关键结构更接近 higher-level slot，而不只是 local slot。
- 只保留 same higher-level slot 的 KV，next-token accuracy 几乎不下降，说明模型真正依赖的 retrieval bucket 接近 higher-level feature。
- Attention output / hidden representation 中可以线性读出较强的 ground-truth feature signal。
- MoE routing 与 attention 捕捉到的 feature 有一定相关性，但 selectivity 仍然不够干净。

这一轮确认：问题不在于 attention 完全没有学到 feature；相反，attention 已经提供了可用 feature signal。真正的问题是 MoE gate / expert 没有把这个 signal 转化为稳定 routing bucket。

## Round 3：Ground-truth Routing 与 Gate Selectivity

文件：[inverse_kv_round3_plan.md](./inverse_kv_round3_plan.md)

结论：

- 直接按照 ground-truth higher-level slot 做 routing 能提升 next-token prediction，说明 feature-based expert bucket 是合理目标。
- Offline probe 说明当前表征中存在可读 feature，gate 并非完全没有能力读出 ground-truth routing。
- Supervised gate 即使达到约 97%/98% routing accuracy，next-token performance 仍接近 baseline，低于 ground-truth dispatch。
- True-router indexing 显示 expert routing 作为 KV reverse index 有潜力；但 early-proxy indexing 失败，说明当前架构还不能直接在同层 attention 前用 `v` 近似 gate input 做可部署 reverse indexing。

这一轮确认：feature-based routing 本身有价值，但 learned gate 的问题不是简单分类准确率，而是训练过程中的 ownership formation。

## Round 4：Baseline MoE 到底按什么分发，以及如何构造可用于 Reverse Indexing 的 Routing Objective

文件：[inverse_kv_round4_plan.md](./inverse_kv_round4_plan.md)

这一轮不再把 naive inhibition 作为主线。已有结果显示，naive inhibition 主要让 routing 变 sharp 甚至 collapse，不能直接证明模型形成了有意义的 feature specialization。

Round 4 只回答两个问题：

1. **Baseline MoE 没按 ground-truth slot 分，那它到底按什么分？**
   这一部分会把 expert assignment 与 local slot、higher-level slot、token id、slot 内位置、boundary、target token、feature frequency、attention cluster 和 representation cluster 做统一相关性分析。

2. **能否用 attention-derived signal 训练一个 pre-attention routing，使 expert bucket 对齐 attention retrieval bucket？**
   这一部分会用 attention mass coverage 构造 positive set，而不是用固定 token 数量。目标是训练一个 attention 前就能产生的 routing signal，使它可以真正用于同层 KV reverse indexing。

这一轮确认：下一步重点不是继续问 inhibition 是否有效，而是先解释 baseline routing 的真实依据，再设计一个显式对齐 attention retrieval bucket、且具备 anti-collapse 约束的 routing objective。

## 当前主结论

1. **Attention 已经学到 hierarchy feature。**
   Higher-level slot 是当前 synthetic hierarchy task 中更接近真实 retrieval bucket 的结构。

2. **Baseline MoE 没有自然形成足够好的 expert selectivity。**
   标准 next-token loss 下，gate / expert 不会自动把可读 feature 转化成稳定 expert bucket。

3. **Ground-truth feature routing 是有价值的。**
   直接按 higher-level slot routing 能提升 NTP，因此 feature-based expert bucket 不是错误目标。

4. **Gate accuracy 不等价于 expert ownership。**
   训练后期监督 gate 到 97%/98% accuracy，仍不能复现 ground-truth dispatch 的收益。

5. **Reverse indexing 有潜力，但当前可部署路径还没打通。**
   真实 gate routing 能筛掉大量 KV 且保持较高 accuracy；但 attention 前 proxy 不能可靠预测真实 gate routing。

6. **下一步关键是解释 baseline routing，并学习可部署的 pre-attention routing。**
   Round4 的核心是先回答 baseline MoE 到底按什么分发，再用 attention-derived signal 训练能够服务 reverse indexing 的 routing bucket。

## 推荐阅读顺序

1. 先读本文，理解整体故事线。
2. 再读 [inverse_kv_round1_notes.md](./inverse_kv_round1_notes.md)，确认 baseline 问题。
3. 再读 [inverse_kv_round2_notes.md](./inverse_kv_round2_notes.md)，确认 attention 已经有 feature signal。
4. 再读 [inverse_kv_round3_plan.md](./inverse_kv_round3_plan.md)，理解为什么问题转向 gate / expert ownership。
5. 最后读 [inverse_kv_round4_plan.md](./inverse_kv_round4_plan.md)，跟进 baseline routing diagnostic 和 attention-derived routing objective。
