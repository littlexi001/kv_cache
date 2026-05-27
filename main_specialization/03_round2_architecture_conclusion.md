# Round2 Architecture Conclusion

与一阶段结论一致：

1. Attention 会自发捕捉到我们定义的 feature：同 slot，无论是 same-input-different-output 还是 different-input-same-output，attention score 的集中度约 80%；
2. Q/K 表征做分发时，即使不加任何约束，也有 60% 的同 feature token 被分发到同一 expert；zipf 数据上，80% 的数据被分发到 2 个 expert 上。

Round2 当前结论：

1. **Gate input representation：** `k` 仍然是最强 router input。`moe-rfull-k-eresid` 达到 `NTP=89.56%`、`same-input-different-output group same-expert=66.09%`、`different-input-same-output group same-expert=66.32%`，明显优于 ordinary hidden router。这说明 attention key 表征更接近 retrieval / output-side feature，而不只是 residual stream 中的 surface token identity。

2. **Gate granularity：** `k/head` 对 different-input-same-output 更强，`B group same-expert` 可到 `75.57%`；但 `full-k` 对 A/B 两类更平衡。因此不能简单说 head-level 全面优于 full-token routing，而应区分目标 feature 类型：如果更关心 output-side feature，`k/head` 更好；如果希望 A/B 两类分布式 feature 都更稳，`full-k` 更合适。

3. **Expert input representation：** expert 输入仍应优先使用 full token vector / `attention_output + residual`。head/head 结构可以提高 routing-attention 对齐，但通常把 NTP 降到 `86%~88%`，说明只让 expert 看 head 子空间会损伤表达能力。

4. **Expert input shape：** expert 使用完整 hidden state 仍然更稳。head/head 结构能提高 attention-expert mass，例如部分结构可到 `80%+`，但整体 NTP 低于 full expert input。因此，当前推荐是 router 可以更 feature-selective，但 expert 尽量处理 full residual token vector。

5. **Regularization：** reused-token 数据显示自然训练仍不能得到很硬的 distributional specialization。后续 regularization 应从 local/high slot 对齐进一步升级为 distributional feature 对齐，例如约束 same-input-different-output group、different-input-same-output group 或 next-token logits 相似的 token 进入相同/相近 expert bucket。

Round2 把 Round1 结论从“head-level 更好”修正为“router input 用 `k` 最稳定，router granularity 取决于希望对齐哪类 feature”。
