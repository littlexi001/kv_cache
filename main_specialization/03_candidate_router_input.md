# Candidate Design: Router Input

Router input 决定 gate 依据什么信息判断当前 token / context 应进入哪个 expert bucket。

1. **Residual / hidden state：** 使用标准 residual stream 或 layer-normalized hidden state 作为 gate 输入。这是 ordinary MoE 的默认方案，预测能力通常稳定，但容易捕捉 token id、target token 或局部 shortcut。
2. **Attention output without residual：** 使用去掉 residual 的 attention output 做 routing。直觉是减少 residual stream 中 token identity 的 domination，让 gate 更依赖 attention 聚合出的上下文信息。但已有结果显示，pure attention output 往往会伤 NTP，且不一定带来稳定 specialization。
3. **Layer input：** 使用 attention 前的 layer input 做 routing。它更接近 pre-attention routing，部署上更友好，NTP 表现也较稳定，但 specialization 通常不如 `k/head`。
4. **Q / K / V：** 使用 attention projection 后的 query、key 或 value 表征做 routing。其中 `k` 当前最值得关注：它本身就是 attention retrieval 中用于匹配历史 token 的表示，因此更可能与 feature bucket / retrieval bucket 对齐。
5. **Pre-attention routing input：** 为了服务 KV cache reverse indexing，routing signal 最好能在 attention 计算前得到。因此，layer input、q、k、v 是比 attention output 更可部署的候选。
6. **Spectral / SVD representation：** 将 hidden states 投影到 SVD / PCA basis 后，在谱空间中做 routing。它试图让不同频率或不同抽象层次的 feature 落到不同子空间，但当前实现中 SVD basis 对 batch 采样和训练动态敏感，效果不稳定。
