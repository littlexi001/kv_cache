# Value-mediated causal closure v2：16K、8-seed 复核

## 结论

**8K 的局部 `score -> Value/residual -> answer margin` 闭环在 16K 独立复现。** 这把机制证据从“单长度偶然相关”推进为 8K/16K 两个长度均稳定，但仍不是检索方法或新位置编码的性能结果。

| 长度 | 子集 | events | Pearson | seed-cluster 95% CI | Spearman | 95% CI | 符号正确率 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 8K | Gold + conflict target | 16 | 0.960 | [0.938, 0.993] | 0.947 | [0.850, 1.000] | 93.75% |
| 16K | Gold + conflict target | 16 | **0.973** | **[0.946, 0.988]** | **0.976** | **[0.910, 1.000]** | **93.75%** |
| 16K | Matched-random evidence | 16 | 0.051 | [-0.605, 0.508] | -0.047 | [-0.655, 0.460] | 56.25% |
| 16K | 全部 target 类别 | 32 | 0.866 | [0.749, 0.935] | 0.820 | [0.704, 0.893] | 68.75% |

16K evidence-only 的回归斜率为 `1.160`（95% CI `[1.031, 1.321]`），截距为 `0.0083`（`[-0.0119, 0.0285]`），$R^2=0.946$（`[0.896, 0.976]`）。截距覆盖 0，斜率接近但略高于 1：局部一阶式几乎给出有校准的有限干预预测，同时 `+0.25` score 已表现出少量高阶放大。

## 真正有价值的区分

先在每个 seed 内对 layer/head/token 聚合，再以 seed 为独立单位 bootstrap：

| Gold − conflict 指标 | 16K 差值 | seed-cluster 95% CI |
|---|---:|---:|
| $\partial m / \partial s$ | **1.284e-3** | **[0.815e-3, 1.861e-3]** |
| Direct centered-OV margin derivative | **0.02632** | **[0.02185, 0.03018]** |
| Baseline attention probability | 0.00050 | [-0.00164, 0.00275] |
| RoPE suppression gap | 0.0939 | [-0.0257, 0.2319] |

它复现了 8K 的核心观察：

> Gold 与 plausible conflict 获得的 attention probability 和 RoPE suppression 没有可分离差异，但它们通过 Value/残差路径对正确答案 margin 的平均有向作用显著不同。

所以“把被 RoPE 压低最多的 token 修回来”不是充分方法；必须估计该具体 score coordinate 的下游 causal utility。

## 协议与限制

- Qwen3-8B，未量化 BF16，16,384 tokens，seed 0–7。
- 仅使用远程物理 GPU 6/7；每卡 4 seeds。
- 每个 seed 对 gold、conflict、lexical、filler 各干预一个 oracle-selected score，并配同 layer/head/class 的随机位置。
- 每次 score 增量为 `0.25`；所有 delta 相对完全同代码路径的 `epsilon=0` no-op。
- 8/8 replay/prefix 审计通过；instrumented 与 no-op 的 margin/NLL 差均严格为 0。
- custom no-op 相对 native 的最大绝对 pair-margin drift 为 `0.2388`，但 8/8 top-1 决策一致；结论只针对 decision-preserving custom graph 内的因果闭环。
- 目标排名使用真实答案 margin gradient，是 oracle diagnostic，不能部署。
- target 与 random 的 decisive-token 身份不匹配，不能把二者的 PPL/accuracy 当成公平方法对比。
- 32K BF16 在 24GB 3090 上因 native eager `repeat_kv` OOM，尚无 32K 闭环结果。

统计采用 50,000 次 percentile seed-cluster bootstrap，固定 RNG seed `20260801`。完整可复现口径见 `independent_seed_cluster_audit.json/.md`。

