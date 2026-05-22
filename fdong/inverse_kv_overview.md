# Inverse KV Overview


我们想做 specizalization，认为做到了就有一万个好的下游性质：1 预测&端侧；2 kv；3 遗忘；
怎么做：
1. 如何定义 specialization
   1. synthetic：生成序列数据，同一个 local slot 里面应当是同一个 feature，发到同一个 expert
   2. real corpus：next token logits 一样的是同一个 feature，发到同一个 expert
2. 为什么现在的 MoE 没做到 specialization
   1. 现在的 MoE 如何分发？：HRJ，syn & real
      1. 同一个 expert 的 token 有啥关系
      2. 不同 expert 的 token 有啥关系
   2. 理想情况下的 specialization：CAR，真实数据
      1. 同一个 expert 的 token 有啥关系
      2. 不同 expert 的 token 有啥关系
   3. 模型各种结构/训练因素对 specialization 有何影响 ZX: 真实数据
      1. load-balance loss -
      2. 残差链接 (oracle)
      3. attention：attention 已经捕捉到 near ground truth -(oracle, syn) ZX: real
3. 什么结构能做到 specialization，并且有助于我们的三个下游任务：DF, LYM synthe; LET: real
   1. oracle：去掉残差
   2. SD on forward representation：
   3. head-level
   4. hierarchical
   5. 为了服务于最后的 kv cache，moe input 不能是 attention output，可以是 layer input ,q ,k, v 

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

- Attention 对 ground-truth slot feature 有明显敏感性，有 75% 的 attention score 集中在 same-slot 中。
- 标准 MoE routing 基本没有自然形成和 local slot / higher-level slot 对齐的 expert bucket。

这一轮确认：标准 Attention 能看到 feature，但标准 MoE 不会自然给出我们想要的 feature-level routing。

## Round 2：Attention 捕捉的 feature 是否已经提供接近 Gound-Truth Signal

文件：[inverse_kv_round2_notes.md](./inverse_kv_round2_notes.md)

结论：

- Attention 捕捉到的关键结构更接近 higher-level slot，已经提供了可用 feature signal.

这一轮确认：问题不在于 attention 完全没有学到 feature；相反，attention 已经提供了可用 feature signal。真正的问题是 MoE gate / expert 没有把这个 signal 转化为稳定 routing bucket。

## Round 3：Ground-truth Routing 对模型有好处吗？

文件：[inverse_kv_round3_plan.md](./inverse_kv_round3_plan.md)

结论：

- 有好处。直接按照 ground-truth higher-level slot 做 routing 能提升 next-token prediction，说明 feature-based expert 是合理目标。
- Supervised gate 达到 98% routing accuracy 时，expert routing 作为 KV reverse index 有潜力 能实现 35% 的 kv_cache 只损失 1.5% next-token-prediction 精度。

这一轮确认：feature-based routing 本身有价值，但 learned gate 的问题不是简单分类准确率，而是训练过程中的 ownership formation。

## Round 4：Baseline MoE 为什么没学到 ground truth feature? 

文件：[inverse_kv_round4_plan.md](./inverse_kv_round4_plan.md)

子问题 & 结论：
1. **Baseline MoE 没按 ground-truth slot 分，那它到底按什么分？**
   1. 同一 input token id, 75% 会被分发进几乎同一 expert 中。
   2. 输入同一 expert 的不同 token，关系是：
   3. 输入不同 expert 的不同 token，关系是：

2. **能否用 attention-derived signal 训练一个 routing，使 expert load 对齐 attention score？**
   可以。
   1. 约束发到同一 expert 的 token 相互 attention score 高，不同 expert 的 token 之间 attention score 低，能实现 expert 上的 token 和 attention score 80% 匹配。
   2. 这种分发与 ground truth slot 的匹配程度达到 95%。


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
