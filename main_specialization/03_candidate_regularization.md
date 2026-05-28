# Candidate Design: Regularization

自然训练得到的 specialization 仍然不够强。即使当前最好的 synthetic 结构也只能达到中等强度的 local / high-slot 对齐，因此需要显式 regularization 或 supervision。

1. **Attention-derived routing objective：** 让 attention score 高的 token pair 具有更相似的 router logits 或更高的 expert-overlap。这个方向最直接服务 reverse KV：如果 gate bucket 能预测 attention retrieval bucket，就可以用 routing bucket 做 KV reverse index。
2. **Logits-similarity regularization：** 在 synthetic 数据上，可以约束同 local / high slot 或高 attention mass token 的 router logits cosine similarity 更高，不相关 token 的 router logits 更低。
3. **Next-token-logits regularization：** 在真实数据上，可以约束输入同一 expert 的 token position，其最终 next-token logits 分布更相似。这对应 real-corpus feature 的 downstream behavior 定义。
4. **Load-balance loss：** 保证 expert usage 不 collapse，但它本身不是 specialization objective。它应作为稳定训练的辅助项，而不是核心目标。
5. **Common expert / top-k routing：** common expert 和 top-k 可以提升 NTP，但会让 hard specialization 指标变软。若目标是 reverse KV，top-1 routing 通常更干净；若目标是预测性能，top-k 和 common expert 可能更有价值。
6. **Ground-truth routing / supervised routing：** 在 synthetic 数据上可作为 upper bound 或 diagnostic，验证 feature-based routing 是否本身有收益；但它不是最终可部署方案。
