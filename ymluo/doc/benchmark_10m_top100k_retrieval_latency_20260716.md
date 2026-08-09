# 10M 文本检索 100K tokens 的时间估计

> **一句话结论：** 在 10M tokens 中返回约 100K tokens，BM25、E5 RAG 和 Hybrid RAG 的在线检索中位时间分别约为 **2.61 ms、7.67 ms 和 18.23 ms**；当前实现的 Full QK 和 PCA64+INT4 在索引已驻留 GPU 时分别约为 **0.777 s 和 0.955 s**，而 PCA 的主要收益是把索引从 **9.54 GiB** 压到 **1.27 GiB**，当前并没有带来扫描加速。

## 1. 测试口径

- 外部记忆：10,000,000 tokens。
- `block_sz=64`，共约 156,250 个 block。
- 返回预算：`ceil(100000 / 64) = 1563` 个 block，即最多 100,032 tokens。
- 在线时间只统计 query 编码、全局打分和 Top-K/融合，不统计离线建库。
- 不统计取回 100K tokens 后的 reader prefill 和答案生成；那是另一项成本。
- BM25/E5 使用真实 XSum BBC 新闻组成的 10M tokens；QK 使用 Llama-3.1-8B 在真实 LongMemEval 文本上提取的 1,024 个 block profile，不是人工高斯向量。

## 2. 主要结果

| 方法 | 10M 全局扫描/打分 | 当前在线总时间，中位数或外推 | 离线建库 | 索引大小 |
|---|---:|---:|---:|---:|
| BM25 | 已包含在总时间中 | **2.61 ms** | 18.96 s | 128.0 MiB RAM |
| E5 Dense RAG | 0.384 ms | **7.67 ms** | 42.26 s | 228.9 MiB |
| Hybrid BM25+E5 RAG | BM25 + E5 | **18.23 ms** | 串行约 61.2 s | 约 357 MiB |
| Full QK | **112.9 ms** | **777.3 ms**，索引驻留 GPU | K 提取约 53.8 min/单卡；理想 8 卡约 6.73 min | **9.54 GiB** |
| PCA64+INT4 | **291.0 ms** | **955.4 ms**，索引驻留 GPU | K 提取同上；之后投影与打包外推约 0.37 s，不含 PCA 基底拟合 | **1.27 GiB** |

这里将 E5 Dense 作为标准 embedding RAG baseline；Hybrid 是此前使用的 BM25+E5 融合版本，因此一并列出。Hybrid 的 8 个 query 中出现了一次约 370 ms 的 Python 暂停，所以表中采用更稳定的中位数；其余 7 次约为 18 ms。

## 3. QK 时间为什么这么长

当前 Full QK 的 777.3 ms 由三部分组成：

| Full QK 阶段 | 时间 |
|---|---:|
| 从当前生成状态提取 query Q | 34.8 ms |
| 扫描全部 156,250 blocks | 112.9 ms |
| CPU selected-head + Top-K + RRF | **629.6 ms** |
| 合计 | **777.3 ms** |

PCA64+INT4 只把第二项换成 291.0 ms，因此总时间变成 955.4 ms。真正的第一瓶颈不是 QK 点积，而是当前实现把 8 层 × 32 个 query heads 的 256 个通道都打分后，在 CPU 上选 16 个通道、逐通道取 Top-1563，再用 Python RRF 融合。这一段占 Full QK 总时间约 81%。

若将选头、Top-K 和融合写成 fused GPU kernel，粗略工程下限可能是：

- Full QK：约 0.15--0.17 s/query。
- PCA64+INT4：约 0.33--0.35 s/query。

这两个数字是基于已测 Q 提取和扫描时间的工程推测，不是实测结果。

## 4. 为什么 PCA64+INT4 反而比 Full QK 慢

这不是 PCA 数学计算量更大，而是两个路径的内核成熟度不同：

- Full QK 走 FP16 矩阵乘，RTX 3090 的 Tensor Core 和 PyTorch GEMM 已高度优化。
- 当前 PCA 内核需要解包 INT4、乘 scale，并为多个 query tail token 和通道写出大块 FP32 分数。
- 当前内核没有把解量化、通道选择、跨 token 聚合和 Top-K 融成一次流式扫描，因此节省的带宽没有转化为延迟收益。

所以当前结论应是：**PCA64+INT4 是存储压缩方案，不是已经验证的加速方案。** 它将索引压缩约 7.5 倍，但扫描约慢 2.58 倍。

## 5. 3090 上的实际部署含义

上述 0.777 s 和 0.955 s 都假设索引已在 GPU 中：

- Full QK 索引为 9.54 GiB。Llama-3.1-8B FP16 权重约 16 GiB，加上运行时缓存后，无法与 Full 索引共同放进单张 24 GB 3090。可以把索引放到另一张 GPU；若每题从 CPU 传约 10.24 GB，按 8--12 GB/s 有效 PCIe 带宽估计还要增加约 0.85--1.28 s，总时间约 1.6--2.1 s/query。
- PCA64+INT4 索引为 1.27 GiB，更可能与模型共存于一张 3090，因此 0.955 s 是更接近可部署状态的当前估计。
- 如果没有预先建立 K/PCA 索引，而是查询时重新对全部 10M tokens 做 Llama 前向，外推为约 53.8 分钟/单卡；即使理想 8 卡线性并行也约 6.73 分钟。这个路径不可用，预计算是必要条件。

## 6. 小样本外推是否可信

QK 扫描只真实 profile 了 1,024 个 block，然后将这些真实 profile 重复到 8,192 blocks 做纯吞吐计时；重复数据不参与任何质量指标。扫描时间如下：

| blocks | Full QK | PCA64+INT4 |
|---:|---:|---:|
| 1,024 | 0.798 ms | 1.843 ms |
| 2,048 | 1.456 ms | 3.766 ms |
| 4,096 | 2.997 ms | 7.793 ms |
| 8,192 | 5.907 ms | 15.187 ms |

2K--8K 区间接近线性。按各采样点独立外推，Full QK 的 10M 扫描范围为 111.0--114.3 ms，PCA64+INT4 为 287.4--297.3 ms，因此“纯扫描时间”的外推较稳定。CPU RRF 和跨设备传输不适合用该线性关系推断，已单独计时或单独估算。

## 7. 对当前项目的直接判断

1. 若目标只是从 10M 中快速取回 100K tokens，BM25/E5 RAG 在速度上已经占绝对优势，QK 全局扫描暂时不具备系统优势。
2. QK 更合理的位置是 L1 原生细排：先由 BM25/E5/轻量 scope index 将 156,250 blocks 缩到几百个候选，再用模型原生 QK 判断“当前生成步骤真正需要什么”。
3. 下一项值得做的工程不是继续提高 PCA 维度，而是实现 GPU fused `head gate + streaming Top-K + fusion`，移除约 630 ms 的 CPU 聚合。
4. 时间测试不代表质量。后续必须在同一个候选预算和 reader 预算下同时报告 evidence recall、Answer@48、检索延迟和端到端延迟，才能判断模型原生细排是否提供了 RAG 没有的增益。

原始计时证据：`ymluo/doc/1b_context_search_research_exploration/evidence/benchmark_10m_top100k_retrieval_latency_20260716.json`。

