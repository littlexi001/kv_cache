# One-shot Risk 组合实验：质量与速度

更新时间：2026-07-20

## 1. 比较的方法

本轮固定候选池为 8%，PCA 索引为 post-RoPE PCA64 + log-scale INT4，最终均使用原始 K 做 exact rerank。

| 方法 | PCA 扫描 | 最终 attention 预算 | V attention |
|---|---|---|---|
| Full KV | 不检索 | 100% | Full SDPA |
| 当前主方法 | 每步完整 PCA64 | 固定每 query head top-2% | Auto-split |
| One-shot + 固定 2% | Expected-Crossing One-shot 动态扫描 16/32/48/64 维 | 固定 top-2% | Auto-split |
| One-shot + 动态预算 | 同上 | 每 head 从 0.5/1/2/3/4/6/8% 中选最小安全预算 | Auto-split |

One-shot 不训练 router。它根据当前 query 相对上一时刻 score cache 的 top-k crossing risk，决定本步需要更新多少个 PCA 频带。

## 2. 实验协议

- 模型：Qwen3-4B-Instruct，FP16。
- 硬件：RTX 3090。
- PPL：128,000-token history；Medicine、Politics、Computer、Space；每主题 1,024 个目标 token。
- Decode：online 时间不含 prompt prefill，包含 2,047 次单-token model forward、首次 PCA basis/INT4 索引构建、检索、exact QK 和 V attention。
- 子模块：真实 128K Medicine trace，Layer 16，连续 16 个真实 query；包含检索、风险规划、top-k、exact QK 和 V attention，不含 MLP。

## 3. PPL 质量

| 方法 | 四主题 mean NLL | 几何 PPL | 相对 Full 质量保持率 | Attention links | 平均 PCA 扫描 |
|---|---:|---:|---:|---:|---:|
| Full KV | 2.34451 | 10.42814 | 100.00% | 100% | -- |
| 当前主方法 | 2.33769 | 10.35732 | 100.68% | 2.000% | 100% |
| One-shot + 固定 2% | **2.33764** | **10.35671** | **100.69%** | 2.000% | 52.75% |
| One-shot + 动态预算 | 2.36803 | 10.67633 | 97.68% | 1.494% | 54.00% |

固定 2% 的 One-shot 与当前主方法质量等价。动态预算中约 75%--78% 的 head 选择 0.5% 动作，预算过于激进，导致四主题 PPL 明显退化。

## 4. Attention 子模块速度

| 方法 | 完整流水线耗时 / layer / token | 相对 Full SDPA | 索引状态 / Full FP16 K+V |
|---|---:|---:|---:|
| Full SDPA | 2.396 ms | 1.000x | 0% |
| 当前主方法 | **0.893 ms** | **2.685x** | 7.17% |
| One-shot + 固定 2% | 1.288 ms | 1.861x | 8.76% |
| One-shot + 动态预算 | 2.801 ms | 0.856x | 8.76% |

One-shot 虽然少扫描约一半 PCA 维度，但增加了 score cache、一次风险规划和多个 masked band kernel。当前 CUDA 实现中，这些开销大于少扫 PCA 维度的收益。

## 5. 整模型 Decode 速度

Medicine + Politics 的固定预算配对结果：

| 方法 | Online 时间 | 相对 Full |
|---|---:|---:|
| Full KV | 1,143.75 s | 1.000x |
| 当前主方法 | **457.39 s** | **2.501x** |
| One-shot + 固定 2% | 633.44 s | 1.806x |
| One-shot + 动态预算 | 886.51 s | 1.290x |

动态预算在 Politics 上的 PPL 为 12.482，差于 Full 的 12.321 和当前主方法的 12.164。

Computer 和 Space 的动态预算 PPL 使用补 Auto-split 前的 ragged V kernel；Auto-split 只改变并行 reduction，不改变选中 token 和 attention 权重。速度只统计补齐 Auto-split 后重新运行的 Medicine 和 Politics。

## 6. 结论

1. 当前默认版本保持不变：完整 PCA64 + 8% candidate + exact rerank + 固定 per-head top-2% + Auto-split。
2. One-shot + 固定 2% 的数值质量成立，但它把子模块加速从 2.685x 降到 1.861x、decode 从 2.501x 降到 1.806x，不应进入默认速度路径。
3. One-shot + 动态预算同时损失质量和速度，不在当前 Pareto 前沿。
4. One-shot Risk 适合保留为论文中的机制消融：它证明约一半 PCA 维度足以维持固定 top-2% 的 PPL，但要产生实际速度收益，需要把 16/32/48/64 维更新融合成一个持久化 kernel，避免 planner 和多次 launch。

## 7. 复现入口

- 128K 组合 PPL：`scripts/run_oneshot_128k_combination_20260720.sh`
- 真实 trace 子模块测速：`src/benchmark_oneshot_combinations.py`
- 结果：`artifacts/20260720_oneshot_combinations/`
