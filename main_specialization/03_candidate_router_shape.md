# Candidate Design: Router Input Shape

Router input shape 决定 gate 是对完整 token 表征做一次分发，还是对表征的不同子空间分别分发。

1. **Full-token routing：** 对完整 hidden vector 做一次 routing。实现简单、表达稳定，但容易把多种 feature 混在一个 gate 决策里。
2. **Head-level routing：** 将 router input 按 attention head 切分，每个 head 单独 routing。当前 synthetic 结果显示，head-level routing 通常比 full-token routing 更利于 specialization，尤其是 `k/head`。
3. **True head/head MoE：** router 和 expert input 都按 head 切分。它能增强 feature bucket 与 attention retrieval bucket 的对齐，但会明显伤 NTP，说明只让 expert 看 head 子空间会损失表达能力。
4. **Spectral-band routing：** 将表征按 SVD / PCA 方向切成不同 spectral bands，并对不同 band 分别 routing。它理论上接近 hierarchical / feature-subspace ownership，但当前结果显示稳定性不足，暂时不是主线。
