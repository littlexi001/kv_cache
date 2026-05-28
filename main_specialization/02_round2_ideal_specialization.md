# Round2 Ideal Specialization: Reused-Token Synthetic

在 reused-token synthetic 中，理想 specialization 不再是简单的“同 token 同 expert”或“同 slot 同 expert”。更准确的定义是：expert assignment 应当与 future-distribution feature 的相似性单调相关。

具体到当前 controlled reused-token 数据：

1. **same-input-different-output group 应当有较高 expert overlap：** 因为它们是同一个 input state 的 next-token distribution 的不同观测；
2. **different-input-same-output group 应当有较高 expert overlap：** 因为不同 input state 共享相同 output-side projected feature；
3. **token same-expert 不是充分证据：** token-level purity 高可能只是 surface-token specialization；
4. **target same-expert / A-B group same-expert 更接近我们关心的 distributional specialization。**

当前实验观察：

1. 同一 expert 拿到的 token 其表征 cosine similarity 高；
2. 数据服从 zipf 分布后，同一 expert 拿到的 token，其所含 feature 在整体 feature 中的频率位次接近。
