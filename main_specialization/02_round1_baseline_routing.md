# Round1 Baseline Routing: Clean Synthetic

问题：baseline MoE 到底按什么分发？

已有结论是：baseline MoE 没有自然按照 ground-truth local slot / higher-level slot 分发，而是更接近 token id、target token 或局部 shortcut。

需要保留的核心证据包括：

1. 同一个 token id 大约有 70% 会被分到同一个 expert；
2. 同一个 expert 内部的表征向量 cosine similarity 更高，约为 0.5；
3. 同一个 expert 内部的 next-token logits cosine similarity 只有约 0.1，说明 token-id / representation shortcut 强于真正的 downstream-behavior feature；
4. synthetic 数据中，gating 不主要由表征空间 SVD 头部子空间解释，而更依赖 5% 到 20% 的中间子空间。这可能说明 clean synthetic 里的 common feature 还不够接近真实语料中的 high-frequency feature。

部分 synthetic 结论见 [Round4 问题1：Baseline MoE 到底按什么分发](../fdong/inverse_kv_round4_plan.md#问题-1baseline-moe-到底按什么分发)。
