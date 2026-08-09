# Strict frozen-reference MPR：8K smoke test

**模型：** Qwen3-8B（NF4/BF16）  
**数据：** 受控英文两跳检索，length=8192，seed=0  
**设备：** 服务器物理 GPU 6  
**性质：** 单样本机制筛查，不是论文主结果

## 结果

| 方法 | Gold PPL | 首 token 正确 | Evidence recall | Evidence mass | Query 时间 |
|---|---:|---:|---:|---:|---:|
| Full attention | 5.6258 | 0 | 100.00% | 5.397% | 0.17 s |
| Exact post-RoPE Top-2% | 3.7996 | 0 | 34.51% | 5.489% | 0.18 s |
| Exact pre-RoPE Top-2% | 4.6777 | 0 | 15.53% | 5.670% | 0.21 s |
| Strict MPR | **3.4217** | 0 | 15.53% | **5.856%** | 237.75 s |
| L2/count-matched random planes | 5.0016 | 0 | 15.53% | 5.688% | 7.61 s |
| Strict MPR + partition preserve | 4.2262 | 0 | 15.53% | 5.718% | 238.42 s |
| Random planes + partition preserve | 3.6424 | 0 | 15.53% | 5.684% | 7.54 s |

相对 exact pre-RoPE baseline，普通 Strict MPR 的单样本 NLL 变化为：

\[
\Delta\mathrm{NLL}=1.2301-1.5428=-0.3127.
\]

这说明在这个样本上，定向选择频率平面比同 $L_2$、同 plane count 的随机频率更有效；但所有方法首 token 都错误，且只有一个 seed，因此不能写成稳定质量结论。

## 约束是否真正满足

| 检查 | 结果 |
|---|---:|
| Frozen support mismatch | 0 |
| 非触发 token 最大变化 | 0 |
| Random-control $L_2$ 最大匹配误差 | $1.11\times10^{-16}$ |
| Random-control plane-count 差异 | 0 |
| 每个触发 token 的频率对数量 | 8 |
| 最大单频率相移 | 0.25 rad |
| 原始 remote trigger 比例 | **51.64%** |
| Solver calls | **12,493** |

这里暴露出一个关键缺陷：它只在 frequency-pair 维度稀疏，却在 token 维度不稀疏。超过一半的远程候选被修改，因此它仍不满足“稀少 suppression event”的方法定义，也导致两个定向 arm 各需约 238 秒。

partition-preserve 在 BF16 后的最大 log-partition 误差分别为 0.0143 和 0.0217，不应称为数值精确保持；正式实验必须同时报告这个误差。

## 不能从该样本推出什么

1. 不能推出 MPR 提高准确率：所有 arm 均未答对。
2. 不能把普通 Strict MPR 与 random 的差异完全解释为“正确频率方向”：两者虽匹配 $L_2$ 和 plane count，但 random arm 实际 QK lift 仅 0.088，而定向 arm 为 0.820。
3. 不能把 partition-preserve control 当作无效：random + partition preserve 也得到 3.6424 PPL，说明分区重心调整、token 间重分配或有限精度效应本身就可能显著改变输出。
4. 不能扩大当前全触发版本：它计算慢且干预范围过宽。

## 下一步决策

只继续 token-capped 版本：在 exact baseline 上，每层、每个 Query head 仅保留 suppression gap 最大的 top-1 / top-4 remote token，再冻结该 trigger plan；normal、random、partition-preserve 全部 replay 同一计划。这样才能同时满足 token 级和 frequency-pair 级的最小干预，并把 CPU 求解量降低约一个数量级。

原始产物：`summary.csv`、`summary.json`、`rows.csv`、`run.log`。
