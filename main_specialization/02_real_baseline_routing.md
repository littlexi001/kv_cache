# Real-Corpus Baseline Routing

真实数据中没有显式 ground-truth slot label，因此需要用 proxy feature 解释 gating。

当前结论是：真实数据里的 gating 更容易被表征空间的 SVD 头部子空间解释，top 5% 子空间可以解释约 90% 的分发结果。

后续需要继续判断：真实 MoE routing 更接近 token identity、representation cluster、next-token logits cluster、frequency / Zipf rank、position shortcut，还是 attention retrieval bucket。
