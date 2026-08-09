# LongBench 1024-token KV 预算对比：AdaKV、SnapKV 与 H2O

**一句话结论：在 1600 条 LongBench 样本上，SnapKV 以 1024-token KV 预算取得 36.44 分，保留 Full Attention 36.58 分的 99.62%，质量与 AdaKV 持平但平均快 2.05 倍；H2O 只有 33.48 分，因此当前最强且最实用的 KV 压缩基线是 SnapKV。**

## 1. 正式结果

| 方法 | LongBench | Full 分数保留率 | 平均耗时 | median | p95 |
|---|---:|---:|---:|---:|---:|
| Full Attention | **36.58** | 100% | 4.780 s | 2.716 s | 16.779 s |
| SnapKV-1024 | 36.44 | **99.62%** | **4.523 s** | **2.798 s** | **16.075 s** |
| AdaKV-1024 | 36.36 | 99.39% | 9.294 s | 4.880 s | 42.194 s |
| H2O-1024 | 33.48 | 91.53% | 7.591 s | 5.743 s | 21.954 s |
| Hybrid+Recent RAG-1024 | 34.05 | 93.08% | 2.829 s | 1.428 s | 12.587 s |

最直接的排序是：

1. **质量：Full ≈ SnapKV ≈ AdaKV > RAG ≈ H2O。**
2. **三种 KV 方法内部的速度：SnapKV > H2O > AdaKV。**
3. **综合质量和运行成本：SnapKV 明显优于 AdaKV 与 H2O。**

Full 和 RAG 来自此前的受控 runner；三种 KV 方法来自同一个 KVCache-Factory runner。因此质量可以逐样本严格比较，但跨 runner 的绝对耗时只作为工程参考。SnapKV、AdaKV、H2O 三者之间的耗时是同框架可比的。

## 2. 公平实验口径

- 模型：Llama-3.1-8B-Instruct，FP16，单样本推理。
- 数据：LongBench 16 个任务，每任务相同的 100 条样本，共 1600 条。
- 上下文：最大 7500 tokens，实际平均 6079 个原文 tokens；平均完整 prompt 为 6272 tokens。
- 生成：greedy decoding，使用相同任务 prompt、最大生成长度和停止条件。
- KV 预算：所有 32 层都采用 1024 个 token slots；Llama-3.1-8B 的 8 个 KV heads 按实际存储粒度计算，而不是把 GQA 重复成 32 个 query heads。
- 共同参数：保留最近 8 个 tokens；其余 1016 个位置由方法评分选出；局部分数使用 kernel size 7 和 max pooling。
- 样本完整性：AdaKV、SnapKV、H2O 各 1600 条，合计 4800 条，样本 ID 完全对齐，无失败或断线重跑。

这里的“1024-token KV 预算”表示每个 KV head 的固定长度容量。SnapKV 和 H2O 在每层的 8 个 KV heads 上各保留 1024 个槽位；AdaKV 使用近似相同的总槽位数，但在 KV heads 之间非均匀分配。它们与 RAG 最终产生的 KV 元素总量大致同阶，但保留位置的结构不同。

## 3. 三种方法实际做了什么

### SnapKV

SnapKV 使用 prompt 末尾最近 8 个 query tokens 对整段历史 K 计算 attention 分数；每个 KV head 独立做局部 max pooling 和 Top-K，保留 1016 个历史位置，再拼接最近 8 个位置。

LongBench 的问题通常位于 prompt 尾部，因此这相当于用“问题对应的内部 query”直接挑证据。不同 KV heads 可以保留不同 token 位置，所以虽然总 KV 槽位数与“所有 head 共用 1024 个文本位置”相同，它们的 token 位置并集可以远大于 1024。

### AdaKV

AdaKV 的 token 评分与 SnapKV 类似，但不会给每个 KV head 固定分配 1016 个历史槽位：

- 每个 KV head 至少获得 20% 的基础容量；
- 剩余约 80% 容量根据各 head 的高分 token 数量全局重分配；
- 不同 head 最终具有不同长度，需要可变长 KV 布局和额外元数据。

本实验中，这种 head 间自适应分配没有带来可测的质量提升，却产生了显著排序、重排和可变长执行开销。

### H2O

H2O 按历史 token 累积获得的 attention mass 选择 heavy hitters，同时保留最近 8 个位置。本实现需要分块计算接近完整的 prompt attention 统计，而 SnapKV/AdaKV 只用末尾小窗口查询历史，因此 H2O 的评分更贵，也更偏向“全局经常被关注的 token”，不一定是当前问题最需要的证据。

## 4. 统计检验

| 配对比较 | 分数差 | 95% CI | sign-flip p | 结论 |
|---|---:|---:|---:|---|
| SnapKV - Full | -0.14 | [-0.54, +0.26] | 0.5109 | 无显著差异 |
| AdaKV - Full | -0.22 | [-0.57, +0.12] | 0.2172 | 无显著差异 |
| H2O - Full | -3.10 | [-3.88, -2.32] | <0.0001 | 显著下降 |
| SnapKV - AdaKV | +0.08 | [-0.21, +0.35] | 0.5797 | 两者质量持平 |
| SnapKV - RAG | +2.39 | [+1.09, +3.70] | 0.0003 | SnapKV 显著更高 |
| AdaKV - RAG | +2.31 | [+0.99, +3.62] | 0.00035 | AdaKV 显著更高 |
| H2O - RAG | -0.56 | [-1.93, +0.78] | 0.4180 | 无显著差异 |

因此不能把 SnapKV 的 36.44 和 Full 的 36.58 解读为“确定损失 0.14 分”。在当前 1600 条样本上，差异小于统计噪声；更准确的结论是 **SnapKV 在 1024 KV 预算下近似无损**。

同理，AdaKV 与 SnapKV 的 0.08 分差异没有意义。选择 SnapKV 的理由不是它确定更准，而是质量持平时执行明显更快、实现更简单。

## 5. 分任务现象

SnapKV 相对 Full 的主要下降：

| 任务 | SnapKV - Full |
|---|---:|
| gov_report | -2.95 |
| qasper | -2.07 |
| trec | -1.00 |
| narrativeqa | -0.56 |
| multi_news | -0.55 |

SnapKV 也在若干任务上高于 Full：SAMSum +2.16、LCC +1.28、RepoBench-P +1.05、MuSiQue +0.66。压缩方法偶尔超过 Full 可能来自过滤干扰信息或任务指标噪声；总体配对检验不显著，不能据此宣称压缩提升了模型能力。

H2O 的损失更系统：16 个任务全部不高于 Full，其中 PassageRetrieval-en -11.58、RepoBench-P -8.82、LCC -4.26。说明全局 heavy-hitter 分数对精确检索、代码局部依赖和当前问题条件下的证据保留不够可靠。

## 6. 为什么 AdaKV 慢、SnapKV 快

相对 SnapKV：

- H2O 平均慢 1.68 倍，median 慢 2.05 倍；主要原因是更昂贵的全 prompt attention 统计。
- AdaKV 平均慢 2.05 倍，p95 慢 2.62 倍；它需要逐 head 排序、跨 head 分配预算、构建可变长 KV 和维护额外元数据。
- AdaKV 在长生成任务上尤其慢：GovReport 为 38.91 s，对比 SnapKV 15.55 s；MultiNews 为 40.90 s，对比 15.33 s。

三种方法的平均峰值 allocated 显存都约为 20.1 GB。这不表示 KV 压缩没有节省显存：峰值发生在完整 prefill 和压缩前，且模型权重、attention 临时张量与 CUDA allocator 占主导。按 Llama-3.1-8B 的 32 层、8 KV heads、head dim 128、FP16 估算，平均 6272-token prompt 的纯 KV 约为 784 MiB，1024-token KV 约为 128 MiB；要测到这约 656 MiB 的稳态差异，应单独记录压缩后 cache tensor 字节数，而不是只看整次推理峰值。

## 7. 与 RAG 的边界

RAG 平均只把约 1002 个原文 tokens 送入模型，因而平均耗时最低，但分数低于 SnapKV 2.39 分。SnapKV 的模型先完整读取平均 6079 个原文 tokens，再压缩内部 KV，所以能利用模型已编码的语义状态，质量更高。

这意味着：

- **SnapKV 解决的是长 prompt prefill 之后的 KV 存储和后续生成问题。**
- **RAG 解决的是 prefill 之前从外部文本中筛选输入的问题。**
- SnapKV 当前结果不能直接证明可以搜索 10M 或 1B 未处理 tokens，因为它仍需先对全部文本做前向计算。
- 对超长上下文，更合理的系统是外层低成本检索减少 prefill，内层使用 SnapKV 保留被读入内容的关键内部状态。

## 8. 研究结论与下一步

1. **后续实验应把 SnapKV-1024 作为主要 KV baseline。** AdaKV 在本设置下没有质量收益且慢约 2 倍；H2O 的质量和速度都更差。
2. **当前结果证明了 head-specific token selection 很强。** 等总 KV 槽位下，每个 head 保留不同 token，比 RAG 所有 head 共用同一批文本位置更有表达力。
3. **它尚未解决 10M/1B 的高效搜索。** 三种方法都需要完整 prefill；真正的新方法必须避免对 10M tokens 做逐 query 的完整前向或完整 attention 扫描。
4. **下一项属性实验应量化不同 heads 的保留集合互补性。** 重点测每层/每 KV head 的 Top-1024 token 集合 Jaccard、并集覆盖率、正确证据落在哪些 heads，以及共享 1024 token 集合造成的质量下降。这直接连接“为什么海量上下文可被分解并行搜索”的核心问题。
5. **显存实验应增加稳态 KV bytes 和 decode 吞吐。** 当前 peak allocated 指标无法体现压缩后缓存大小，也无法区分 prefill 与 decode 的收益。

## 9. 产物

- 对齐 runner：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_kvcache_factory_longbench_aligned.py`
- 7 卡启动器：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/launch_kvcache_factory_b1024_aligned_parallel_20260713.sh`
- 断线守护器：`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/supervise_kvcache_factory_b1024_m100_20260713.sh`
- 统一评分器：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/summarize_kvcache_factory_aligned_comparison.py`
- 本地正式汇总：`tmp/kvcache_factory_b1024_m100/analysis/`
- 服务器原始输出：`outputs/kvcache_factory_aligned_b1024_20260713_m100_v1/`
