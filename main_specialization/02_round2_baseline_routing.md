# Round2 Baseline Routing: Reused-Token Synthetic

问题：当 token id、input prefix、output token 与 future-distribution feature 解耦后，baseline gate 仍然按 token id / surface form 分发，还是开始按 distributional feature 分发？

当前结论：

1. 与 Round1 相同，同一个 token id 大约有 70% 会被分到同一个 expert；
2. 具有同一 feature 的 token 中，50% 的 token 会集中在 10% 的 expert 上；90% 的 token 会集中在 35% 的 expert 上；
3. 数据为 zipf 分布后，gate 的分发被表征中协方差最大的几个方向主导，与真实数据上的模型一致，说明当前 synthetic 数据性质已足以解释真实数据的 gating 行为。
