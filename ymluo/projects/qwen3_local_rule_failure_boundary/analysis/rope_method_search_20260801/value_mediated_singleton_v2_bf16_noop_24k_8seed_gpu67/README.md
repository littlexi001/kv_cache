# Value-mediated causal closure v2：24K、8-seed 复核

## 结论

**局部证据坐标闭环在 24K 第三次复现，但相较 8K/16K 略弱。** 这形成了 Qwen3-8B、8K/16K/24K、每个长度 8 seeds 的连续证据；同时继续否定“attention 或 RoPE suppression 本身就能判断真假/效用”。

| 长度 | Evidence target events | Pearson | seed-cluster 95% CI | Spearman | 95% CI | 符号正确率 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 16 | 0.960 | [0.938, 0.993] | 0.947 | [0.850, 1.000] | 93.75% |
| 16K | 16 | 0.973 | [0.946, 0.988] | 0.976 | [0.910, 1.000] | 93.75% |
| 24K | 16 | **0.936** | **[0.849, 0.972]** | **0.900** | **[0.757, 0.991]** | **81.25%** |
| 24K matched-random evidence | 16 | -0.226 | [-0.688, 0.535] | -0.024 | [-0.549, 0.573] | 62.5% |

24K evidence-only 的回归斜率为 `1.055`（95% CI `[0.846, 1.240]`）、截距为 `-0.0084`（`[-0.0280, 0.0158]`）、$R^2=0.876$（`[0.720, 0.946]`）。一阶预测仍近似校准，但更长上下文下有限 `+0.25` score 干预的高阶误差与 BF16 路径噪声更明显。

## Gold 与 conflict 的结构差异

| Gold − conflict 指标 | 24K 差值 | seed-cluster 95% CI |
|---|---:|---:|
| $\partial m / \partial s$ | **1.617e-3** | **[1.160e-3, 2.161e-3]** |
| Direct centered-OV margin derivative | **0.02693** | **[0.02358, 0.03010]** |
| Baseline attention probability | -0.00136 | [-0.00331, 0.00080] |
| RoPE suppression gap | 0.0422 | [-0.0852, 0.1900] |

在第三个长度上，attention probability 与 suppression 的类别差仍跨 0，而 Value/残差有向敏感度的差稳定为正。三段长度共同支持：

> 位置相位和 softmax 决定“读多少”，但 token 的 Value 经后续网络到底支持 gold 还是 conflict，才决定“多读是否有益”。

## 协议与限制

- Qwen3-8B，未量化 BF16，24,576 tokens，seed 0–7。
- 仅使用远程物理 GPU 6/7；每卡 4 seeds。
- 所有 delta 相对同路径 `epsilon=0` no-op；8/8 replay/prefix 审计通过。
- custom no-op 相对 native 的最大绝对 pair-margin drift 为 `0.4638`，高于 8K/16K，但 8/8 top-1 决策仍一致。结论只针对 decision-preserving custom graph。
- target ranking 使用正确答案 margin gradient；不是可部署 selector。
- 本实验不证明准确率或 PPL 改善，target-random 的配对 $\Delta$NLL 置信区间仍跨 0。
- 32K BF16 在当前 24GB eager-native control 上 OOM，没有 32K 数据。

统计采用 50,000 次 percentile seed-cluster bootstrap，固定 RNG seed `20260801`。完整产物见 `independent_seed_cluster_audit.json/.md` 与 `corrected_provenance.json`。

