# 当前结果

## 实验设置

短任务归因使用 Llama-3.1-8B-Instruct 和 LongBench 三个此前下降最大的任务。每个任务 5 条，严格匹配相同 `task/sample_id`。冻结方法使用动态候选预算、c64 sampled quantile、QK-balanced qMSE 索引、rank-16 INT4 ValueSketch 和 exact FP16 sparse attention，无 Full fallback。

比较中，`alpha=1` 是当前固定尾部补偿；`alpha=0` 保留相同 ValueSketch 索引、相同 deterministic `sortcompact` kernel 和候选，只把尾部补偿权重设为零。

## 结果一：固定尾部补偿导致短任务下降

| 条件 | 三任务宏平均 | 相对 Full |
|---|---:|---:|
| Full | 0.490535 | 100.00% |
| 固定 alpha=1 | 0.453697 | 92.49% |
| 固定 alpha=0 | **0.493824** | **100.67%** |

分任务结果：

| 任务 | Full | alpha=1 | alpha=0 |
|---|---:|---:|---:|
| LCC | 0.771231 | 0.746512 | **0.781332** |
| MultiFieldQA-en | 0.481810 | 0.414117 | **0.481810** |
| QMSum | 0.218565 | 0.200461 | **0.218331** |

读取方式：首先比较 alpha=0 与 alpha=1。候选和 selector kernel 没变，因此 0.040127 的宏平均差值来自尾部 Value 校正，而不是 QK 检索。

另外运行了完全关闭 ValueSketch 的 debug 路径。它与 alpha=0 在 15/15 条样本上预测逐字一致，预测 SHA256 都是：

```text
715f30dd24c9cb6cdf0bfaf7693591a0b459daa9fffda4eaffbec99626f1ee9e
```

窄结论：对这三个短任务，当前 rank-16 INT4 尾部补偿是有害噪声，候选检索本身已经足够。该结果不证明所有长度都应使用 alpha=0。

## 结果二：CUDA 分阶段热点

Qwen3-4B、每 head 约 1280 个候选，CUDA Event 直接测量每个生成 token 在所有层上的累计时间：

| 长度 | Query prepare | Retrieval | Sparse attention | Key append | Value append/index | 已标记合计 |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 2.06 ms | 5.42 ms | 14.89 ms | 2.70 ms | 2.48 ms | 27.55 ms |
| 64K | 2.10 ms | 7.63 ms | 28.04 ms | 1.55 ms | 2.46 ms | 41.78 ms |
| 128K（双卡） | 2.53 ms | 6.76 ms | 25.51 ms | 3.19 ms | 2.93 ms | 40.91 ms |

32K 的整模型稳态时间为 61.63 ms/token，64K 为 61.52 ms/token。128K 使用两张 3090，稀疏方法为 101.66 ms/token，Full 为 299.19 ms/token，即 2.94x；双卡的绝对时间不能与单卡 32K/64K 直接比较。已标记阶段不等于整模型时间，未标记部分包括 QKV/O 投影、MLP、归一化、残差连接和框架调度。

窄结论：在已标记 QKSieve 阶段中，`sparse attention + Value tail reduce` 是最大热点；只优化索引构建不能显著改善稳态速度。

## 结果三：固定删除尾部补偿也不通用

Qwen3-4B 使用同一自然代码流、同一目标 token、同一候选集合和 16 个连续预测位置。PPL 越低越好；相对 Full 质量定义为 `Full PPL / 稀疏 PPL`。

| 长度 | Full PPL | alpha=1 PPL | alpha=1 质量 | alpha=0 PPL | alpha=0 质量 |
|---|---:|---:|---:|---:|---:|
| 32K | 1.240061 | 1.219059 | 101.72% | **1.176501** | **105.40%** |
| 64K | 7.870531 | 7.919292 | 99.38% | **7.325362** | **107.44%** |
| 128K | 1.128709 | **1.131180** | **99.78%** | 1.157261 | 97.53% |

读取方式：32K/64K 中，删除尾部补偿更好；128K 中，完整尾部补偿更好。固定 `alpha=1` 和固定 `alpha=0` 都无法覆盖三个长度。该结果支持“根据当前 head 的尾部信号与噪声连续收缩”，但只有一个文本流和 16 个目标 token，尚不能证明 SURE/Ridge 公式有效。

速度试验中 alpha=0 仍然构建 ValueSketch 索引并进入相同主 kernel，因此它不是“删除 ValueSketch 后”的速度上界；32K、64K、128K 的稳态波动也不单调。正式速度结论必须在自适应系数验证后，用能够真正跳过 alpha=0 head 尾部计算的融合 kernel 重测。

## 结果四：方差收缩不足，block 条件残差稳定有效

真实 Q/K/V 来自 NarrativeQA 32K/64K/128K、LCC 64K 和 QMSum 64K。每个 trace 取第 8、17、26 层，每层 32 个 query head，共 96 个 head-case。候选由同一 QK-balanced qMSE proxy top-k 选出。指标是 attention 输出相对 L2；另把同层所有 head 拼接并通过真实 `W_o`，得到投影后相对 L2。

首先，SURE/Ridge 的平均 alpha 在五个 trace 上均为 0.997–0.999，输出误差只比固定 alpha=1 改善约 0.1%–0.3%。因此“独立零均值残差方差”模型没有解释主要误差，不能据此修改主方法。

在 top-1280 下，block-256 conditional residual d8 相对当前全局 ValueSketch 的逐 head mean 误差变化如下：

| trace | 当前误差 | 条件残差误差 | 相对下降 |
|---|---:|---:|---:|
| NarrativeQA 32K | 0.015874 | 0.008263 | 47.9% |
| NarrativeQA 64K | 0.023548 | 0.012527 | 46.8% |
| NarrativeQA 128K | 0.021963 | 0.013407 | 39.0% |
| LCC 64K | 0.010007 | 0.005146 | 48.6% |
| QMSum 64K | 0.024685 | 0.013293 | 46.1% |

窄结论：主要可修复误差来自 block 和 Key 坐标相关的 Value 残差，不是单纯随机噪声。该模型使用当前请求的闭式统计，不使用任务标签或跨数据训练。

## 结果五：候选减少 25% 后仍优于当前方法

参照固定为 `top-1280 + rank-16 全局 ValueSketch`。下表中的误差比均相对该参照，小于 1 更好；“通过任务”要求 mean、P90 和 `W_o` 投影误差同时满足预先写定的标准。

| top-k | 方法 | 通过 trace | 平均误差比 | 最坏 mean 比 | 最坏 P90 比 | 最坏 W_o 比 |
|---:|---|---:|---:|---:|---:|---:|
| 960 | 全局 ValueSketch | 0/5 | 1.292 | 1.367 | 1.356 | 1.359 |
| 960 | block residual mean | 0/5 | 1.060 | 1.151 | 1.231 | 1.263 |
| 960 | block conditional d8 | **5/5** | **0.712** | **0.806** | **0.850** | **0.965** |
| 768 | block conditional d8 | 4/5 | 0.865 | 0.984 | 1.053 | 1.169 |
| 640 | block conditional d8 | 3/5 | 1.004 | 1.140 | 1.159 | 1.346 |

top-960 是当前唯一跨五个 trace 全部通过的低预算配置。top-768 在 NarrativeQA 128K 的 `W_o` 投影误差为参照的 1.169 倍，因此尽管平均值较好，也不能作为安全配置。

## 结果六：低 rank 边界与可部署量化

直接把 ValueSketch rank 和候选数同时压低会在 NarrativeQA 128K 失败：`rank-8/top-960` 和 `rank-4/top-960` 都只通过 4/5 个 trace，最坏 `W_o` 误差比分别为 1.123 和 1.199。因此不能仅凭四个较容易的 trace 冻结更小配置。

边界复核找到两个跨五个 trace 全通过的配置。下表仍以当前 `rank-16/top-1280` 为参照，小于 1 表示误差更低：

| 配置 | block 统计精度 | 通过 trace | 平均 mean 比 | 最坏 mean 比 | 最坏 P90 比 | 最坏 W_o 比 |
|---|---|---:|---:|---:|---:|---:|
| rank-8/top-1120 | FP16 | 5/5 | 0.703 | 0.805 | 0.829 | 0.975 |
| rank-8/top-1120 | INT8 | **5/5** | **0.704** | **0.805** | **0.833** | **0.974** |
| rank-8/top-1120 | INT4 | 5/5 | 0.822 | 0.893 | 0.987 | 1.046 |
| rank-4/top-1280 | INT8 | 5/5 | 0.674 | 0.772 | 0.760 | 0.918 |

`rank-8/top-1120 + INT8 block 统计` 是当前优先做模型级验证的配置：相对冻结方法，exact 候选数减少 12.5%，全序列 Value code 从 8 Byte/token/head 降到 4 Byte/token/head。INT8 block 统计约增加 0.53 Byte/token/head；按冻结索引的 306 bit/token/head 估算，总辅助索引从约 7.47% 降到约 6.79%。INT4 虽然勉强通过预设阈值，但 NarrativeQA 128K 的最坏 `W_o` 比已到 1.046，安全余量太小，不作为首选。

这些是 attention 输出误差结果，不是 PPL、LongBench 分数或实际 CUDA 加速。允许的结论仅是：FP32 元数据并非结果成立的必要条件，而且存在一个候选更少、Value code 更小的可部署候选配置。

## 结果七：固定索引成本与生成长度的 break-even

一次请求的总时间写成：

```text
T_full(G)   = G * t_full
T_sparse(G) = T_fixed + G * t_sparse
G_break     = T_fixed / (t_full - t_sparse)
```

其中 `T_fixed` 是一次索引准备成本，`G` 是生成 token 数。只有 `t_sparse < t_full` 时才存在有限 break-even。

| 历史长度 | 固定成本 | Full 稳态 | 稀疏稳态 | 稳态加速 | 收回固定成本所需 token |
|---:|---:|---:|---:|---:|---:|
| 32K | 0.3445 s | 87.545 ms/token | 61.634 ms/token | 1.420x | 13.30，向上取整为 14 |
| 64K | 0.3685 s | 163.251 ms/token | 61.519 ms/token | 2.654x | 3.62，向上取整为 4 |
| 128K（双卡） | 0.5502 s | 299.194 ms/token | 101.657 ms/token | 2.943x | 2.79，向上取整为 3 |

读取方式：短输出确实会放大固定成本，但在 64K/128K，固定成本不是主要长期瓶颈；生成 4/3 个 token 后就已经收回。真正困难的是约 8K 以下：当前稀疏单步本身可能不比 Full 快，此时仅增加输出长度无法解决问题，必须降低检索/launch 开销或在该长度区间使用更适合的精确 kernel。

对于不可变的共享前缀，索引可跨问题常驻并复用，此时第二次请求开始不再重复支付完整 `T_fixed`。对于持续增长的 agent KV，需要增量追加和一致性验证，现有结果还不能把“不可变前缀复用”直接外推到任意增长缓存。

## 下一项证据

先对 `rank-8/top-1120 + INT8 block 统计` 做模型级 PPL 和 LongBench 弱项小样本；通过后再实现融合 CUDA kernel，直接测候选减少和 Value code 减半带来的阶段时间。与此同时，索引常驻、增量构建、CUDA Graph 和 kernel 融合属于不改变候选与 attention 公式的速度路径，应与数值方法分开验证。
