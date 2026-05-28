# Project Goal

我们的核心目标是实现 **MoE 模型中的 expert specialization**：不同 expert 应当对输入 token / context 中不同类型的 feature 形成稳定、可解释、可复用的分工。

如果模型能够形成稳定的 feature-level expert specialization，它可能带来多个重要下游性质：

1. 预测与端侧部署；
2. KV cache reverse indexing；
3. 持续学习与遗忘控制。

其中，KV cache reverse indexing 是当前最直接的系统动机：如果 expert bucket 能够对应某类 feature，那么 decode 时当前 token 的 routing bucket 就可以作为索引，优先检索历史上属于相同或相关 expert bucket 的 KV，从而减少无关 KV 的访问。

更长期看，如果 expert bucket 对应稳定 feature，也可能支持更可控的端侧部署、expert swap、持续学习和遗忘控制。
