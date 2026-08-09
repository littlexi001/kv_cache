# CountCap 短上下文优化与固定 PCA 基底实验

更新时间：2026-07-24

## 1. 当前应冻结的实现

当前质量优先的实现为：

`countcap_fullprompt_keypca_direct_qkvfused_qprojscan_prefillindex`

其核心流程为：

1. Prefill 期间按请求估计每层、每个 KV head 的 Key-PCA64 基底。
2. 将历史 K 投影到 PCA 空间，以 INT4 和 16-token 分组 scale 存储检索索引。
3. 每个 decode step 使用 256 个均匀样本估计候选阈值。
4. 在一个 CUDA kernel 中完成原始 query 的 PCA 投影、INT8 量化、样本阈值计算和候选写出。
5. 对约 3%--6% 的候选执行精确 QK，并保留最多 2%、上限 2048 token 的最终 attention。
6. 精确 QK、softmax 和 AV 使用融合消费 kernel。

该版本不使用 router，也不回退 Full Attention。

## 2. Query 投影与阈值扫描融合

旧实现依次调用 `einsum`、绝对值、最大值、量化和候选扫描。新 CUDA 路径直接读取原始 query，在一个流水线中完成：

```text
raw query
  -> PCA48/64 projection
  -> per-head INT8 quantization
  -> 256-point sampled quantile
  -> full INT4 index scan
  -> compact candidate list
```

Query 投影和量化微基准：

| 实现 | 每层时间 | 相对速度 |
|---|---:|---:|
| PyTorch 算子链 | 0.18448 ms | 1.00x |
| 融合 CUDA | 0.02711 ms | 6.80x |

投影最大绝对误差为 `1.526e-5`；99.09% 的 INT8 code 完全一致，其余 code 最大只差 1。

## 3. 整模型速度

Llama-3.1-8B-Instruct、RTX 3090、GovReport、生成 64 token。速度统计包含 query 检索、候选构造、精确 attention 和 decode 中的其他模型计算。

| Prompt | Full KV | 原 CountCap | 当前融合版本 | 相对 Full |
|---:|---:|---:|---:|---:|
| 8K | 2.5789 s | 2.9667 s | 2.5843 s | 0.998x |
| 16K | 3.9381 s | 2.9861 s | 2.6752 s | 1.472x |
| 32K | 6.3927 s | 3.8741 s | 3.8394 s | 1.665x |

结论：

- 8K 已从原 CountCap 的 0.87x 提升到与 Full 基本持平。
- 16K 已从约 1.32x 提升到 1.47x。
- 32K 的主要收益来自稀疏 attention，本轮融合与 query-only 融合速度基本相同。
- 8K 的剩余瓶颈已不是 PCA 扫描吞吐，而是每层固定 kernel launch、Python/HF 调度和非 attention 计算。

## 4. 256 与 128 个阈值样本

128 个样本在微基准中比 256 个样本快约 17%，单个 GovReport 样本也显示 0.7%--2.2% 收益。但跨任务测试发现：

- Qasper 分数从 0.36581 降至 0.34696。
- Musique 的生成长度和输出发生变化。
- 128 样本的阈值方差会改变候选覆盖。

因此主路径恢复并冻结为 256 个样本。

恢复 256 后，在 Qasper、Musique、LCC、PassageCount 的 8 个样本上，融合扫描与未融合的 256-sample 路径逐字预测一致。在线时间分别改善约 2.8%、5.0%、2.0% 和 13.0%；最后一项生成极短，绝对时间很小。

## 5. 固定 PCA 基底

固定基底的目标是消除每个请求约 0.5 秒的 PCA 分解成本。

### 5.1 单提示基底

使用一个 GovReport 提示校准，然后跨任务复用：

| 测试任务 | 索引构建加速 | 总耗时加速 | 与自适应 PCA 输出 |
|---|---:|---:|---|
| NarrativeQA | 12.13x | 1.069x | 一致 |
| HotpotQA | 12.89x | 1.105x | 一致 |
| Qasper | 17.74x | 1.392x | 一致 |
| LCC | 16.98x | 1.147x | 不一致，样本分数相同 |

### 5.2 多提示二阶矩基底

使用 GovReport、NarrativeQA、Qasper、RepoBench-P 的 rank-64 二阶矩平均，再重新求特征向量：

- HotpotQA、Musique 与请求内自适应 PCA 输出一致。
- LCC 输出不同，样本分数从 0.1154 提高到 0.5814。
- MultiNews 输出不同，样本分数从 0.1382 降到 0.1134。
- 索引构建仍可快约 12--18 倍，总耗时快约 1.08--1.17 倍。

固定基底的速度收益成立，但质量漂移尚未解决，因此不能替代当前主方法。

## 6. 被否定的方向

### 6.1 严格 progressive PCA early exit

使用 Cauchy 上界在 16/32 维后尝试安全停止。虽然只需读取约 61%--63% 的 Key index 和约 43% 的 query 投影，但分支、寄存器和额外范数读取使扫描速度仅为原实现的 0.79--0.81x。

### 6.2 相邻 step 候选复用

严格安全证书通过率为 0；周期复用在 8K/16K 不加速，并在 32K 出现明显质量下降。该方向不再作为主线。

### 6.3 单提示或少量提示固定 PCA

它能解决 prefill 索引成本，但目前不能保证跨领域质量，暂不进入默认实现。

## 7. 下一步优先级

1. **CUDA Graph / 静态 decode 工作区。** 当前 8K 已到 Full 附近，继续降低扫描算术量意义很小。应固定候选容量和临时 buffer，捕获每 token 的重复 kernel launch，直接降低短上下文的固定调度成本。
2. **8K--32K 多任务交叉点实验。** 在 8K、12K、16K、24K、32K 上使用相同样本、相同生成长度，至少覆盖 QA、摘要、代码三类任务，报告中位数和置信区间。
3. **无数据或大规模校准的通用投影。** 比较模型权重诱导基底、Hadamard/结构化正交投影和大规模跨域二阶矩基底；只有在独立任务上达到与自适应 PCA 等价质量，才考虑替换。
4. **冻结算法后再跑完整论文表。** 当前不应继续改变 2% attention、3%--6% candidate 和 256 sample 等核心设置，避免系统优化和算法质量同时漂移。

当前最值得立即执行的是第 1 项。8K 的问题已经从“检索计算太多”变成“固定调度成本太高”，CUDA Graph 比继续减少 PCA 维度或样本数更可能带来可兑现的端到端收益。
