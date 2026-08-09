# Direct CountCap 融合算子结果

## 改动

方法保持 Key-PCA、INT4、256 点 sampled-quantile 和约 6% 直接 attention
不变，只优化执行路径：

1. 取消每层 `overflow.any()` 引入的 CPU/GPU 同步。
2. 在 ragged value-attention CUDA kernel 内隐式加入 self token。
3. 删除 self token 所需的 `torch.full`、两次 `cat` 和两次 `scatter`。

新方法名为 `countcap_fullprompt_keypca_direct_fused`。

## 正确性

- 16K、960 个候选的 CUDA 对照：最大绝对误差 `0.0`。
- 8K/16K、32/64 输出长度、每点三次重复：12/12 组的新旧 direct
  生成 token 与完整预测完全一致。
- 当前质量验证只使用同一个 GovReport 长样本，证明实现等价，不代表完整
  LongBench 质量结论。

## 微内核

16K history、每个 head 960 个候选：

| 路径 | 时间 |
|---|---:|
| 旧 self 打包 + ragged attention | 0.260 ms |
| 融合 self ragged attention | 0.141 ms |
| 加速 | 1.84x |

## 整模型结果

所有时间包含 dense suffix、Key-PCA/INT4 建表、阈值检索、真实 QK、
稀疏 value attention 和生成循环。

| Prompt | 输出 | 方法 | Online | 相对 Full | Total | Total 相对 Full |
|---:|---:|---|---:|---:|---:|---:|
| 8K | 32 | Full | 1.327 s | 1.000x | 4.448 s | 1.000x |
| 8K | 32 | 旧 direct 6% | 3.019 s | 0.438x | 5.721 s | 0.777x |
| 8K | 32 | 新 fused 6% | 2.064 s | 0.643x | 4.762 s | 0.934x |
| 8K | 64 | Full | 2.598 s | 1.000x | 5.688 s | 1.000x |
| 8K | 64 | 旧 direct 6% | 4.713 s | 0.552x | 7.376 s | 0.771x |
| 8K | 64 | 新 fused 6% | 3.666 s | 0.708x | 6.332 s | 0.898x |
| 16K | 32 | Full | 2.021 s | 1.000x | 8.765 s | 1.000x |
| 16K | 32 | 旧 direct 6% | 2.967 s | 0.681x | 9.253 s | 0.947x |
| 16K | 32 | 新 fused 6% | 2.079 s | 0.977x | 8.417 s | 1.041x |
| 16K | 64 | Full | 3.988 s | 1.000x | 10.802 s | 1.000x |
| 16K | 64 | 旧 direct 6% | 4.782 s | 0.832x | 11.110 s | 0.972x |
| 16K | 64 | 新 fused 6% | 3.776 s | 1.054x | 10.135 s | 1.066x |

新 fused 相对旧 direct 的 online 加速为 `1.27x-1.46x`。

## 成本模型

使用 `Tonline = Tfixed + (G - 1) * Tstep` 拟合：

| Prompt | 方法 | Tfixed | Tstep | 相对 Full 交叉点 |
|---:|---|---:|---:|---:|
| 8K | Full | 0.097 s | 39.70 ms | - |
| 8K | 旧 direct 6% | 1.379 s | 52.92 ms | 不存在 |
| 8K | 新 fused 6% | 0.513 s | 50.04 ms | 不存在 |
| 16K | Full | 0.115 s | 61.46 ms | - |
| 16K | 旧 direct 6% | 1.208 s | 56.73 ms | 约 232 token |
| 16K | 新 fused 6% | 0.435 s | 53.02 ms | 约 39 token |

## 结论

融合算子显著降低了稀疏方法的固定开销，并使 16K 的 online break-even
从约 232 个生成 token 降至约 39 个。16K、64-token 生成已经达到
`1.054x` online 和 `1.066x` total 加速。

8K 仍然无法超过 Full，因为 fused 路径的单步时间仍为 `50.04 ms`，
高于 Full SDPA 的 `39.70 ms`。下一步需要融合阈值检索、candidate QK
和 value attention，或者为 8K 使用独立的轻量内核；继续优化 self
打包的收益已经接近耗尽。
