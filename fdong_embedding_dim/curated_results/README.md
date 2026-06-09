# Curated Results for Embedding-Dim Experiments

这个目录只保存适合 git 同步和讨论展示的轻量结果。全量 sweep 输出保留在本机 `fdong_embedding_dim/outputs/`，默认不进入 git。

## 结论摘要

1. **提升维度提供的是潜在 residual 表征容量，不会自动改善 tail。**
   如果 tail 初始化时避开 common 主方向，三维空间中的额外 residual plane 会被 tail 使用，tail loss / margin 明显改善。

2. **初始化不好时，raw dimension 提升也可能无效。**
   Packed 初始化会让 tail 继续停留在低 effective dimension 的竞争结构里。此时即使从 2D 提升到 3D，tail 仍然可能只使用一维 residual subspace。

3. **导师的梯度竞争理论和初始化观察可以统一。**
   初始化决定竞争结构是否在训练早期出现；梯度干扰 / effective rank 理论解释这种竞争为什么会影响收敛后的表征空间和预测效果。

4. **Tail 内部会递归出现 head-tail 结构。**
   如果 tail 内部还有 Zipf 分布，tail-high 会获得更低 loss、更高 margin 和更高 SIR；tail-low 最差。

5. **Transformer 结果支持大方向，但不能机械外推 toy 结论。**
   Transformer 中 common/head 仍形成主子空间，tail 主要进入 residual space；但多层 hidden representation 可以把 packed embedding 中的 tail 重新展开。

## 文件组织

```text
figures/2d_trajectories/
  二维训练轨迹图，用于展示 spread vs packed 初始化如何改变最终几何。

figures/3d_trajectories/
  三维训练轨迹图，用于展示 3D 中 tail 是否真的使用新增维度。

figures/3d_residual_planes/
  去掉 common 方向后的 residual plane 图，是解释 effective dimension 的核心可视化。

figures/gradient_interference/
  训练动态机制图，展示 tail gradient effective rank、SIR、gradient cosine 等指标。

tables/
  轻量汇总表和 Transformer spectral occupation JSON。
```

## 关键表

- `tables/toy_gradient_interference_all_runs_summary.csv`
- `tables/transformer_spectral_occupation.json`

## 推荐展示顺序

1. 先看 `figures/3d_residual_planes/tail3_uniform_spread_residual_plane.png` 和 `tail3_uniform_packed_common_residual_plane.png`。
2. 再看 `figures/3d_trajectories/` 中对应的 3D trajectory。
3. 然后看 `figures/gradient_interference/` 中的 spread vs packed 对照。
4. 最后结合 `tables/toy_gradient_interference_all_runs_summary.csv` 讲 effective dimension / SIR / loss 的统一解释。
