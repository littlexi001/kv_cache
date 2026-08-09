# QKSieve 解码冗余操作裁剪：结果

## 实验设置

本页比较 Qwen3-4B-Instruct-2507 在单张 RTX 3090、64K 历史下的不同裁剪版本。公平 Full KV CUDA Graph 延迟为 25.891 ms/token。稀疏路径保持每 head 最多 1,280 个 exact K/V 候选，完整 K/V 和辅助索引都常驻 GPU。

质量使用四条确定性自然文本流，每条 64 个 teacher-forced token。`top-1` 表示稀疏和 Full 下一 token 预测相同的比例。首先看 top-1，再看几何 PPL；单独 PPL 更低不能证明算法更通用。

## 阶段延迟

| 变体 | 64K Graph ms/token | 相对公平 Full | 结论 |
|---|---:|---:|---|
| Full KV | 25.891 | 1.000x | 公平基线 |
| 完整 ValueSketch c64 | 24.689 | 1.049x | 原当前版本 |
| 完整 ValueSketch c32 | 24.452 | 1.059x | 缩小样本有效但收益小 |
| 完整 ValueSketch s512 | 23.877 | 1.084x | 512 样本可保留 |
| Unsorted rank-16+s512 | 25.235 | 1.026x | 取消排序反而更慢 |
| NoVS+s512，自动 split | **21.411** | **1.209x** | 最快候选，但质量安全性不足 |
| NoVS+s512，split 4 | 21.830 | 1.186x | 比自动配置慢 |
| NoVS+s512，split 2 | 24.881 | 1.041x | 并行度不足 |
| NoVS+s512，split 1 | 33.730 | 0.768x | 比 Full 更慢 |
| Mean-tail+s512 | 22.241 | 1.164x | 质量失败 |
| ValueSketch early 12 | 22.074 | 1.173x | 单窗口 top-1 仅 93.75% |
| ValueSketch mid 12 | 22.630 | 1.144x | 四主题一致率不足 |
| ValueSketch late 12 | 22.178 | 1.167x | 四主题一致率不足 |
| ValueSketch mid+late 24 | 22.733 | 1.139x | 一致率仍不足 |

如何阅读：删除阈值样本和 ValueSketch 的确降低了延迟；但减少 exact attention 分块并未省时，因为它损失了 GPU 并行度。因此“少做 kernel”必须以实测延迟判断。

旧 c64 eager profiler 在 64K 上给出的逐 token kernel 阶段为：候选 sparse attention 28.04 ms、代理检索 7.63 ms、ValueSketch 索引 2.46 ms、query 准备 2.10 ms、Key append 1.55 ms。该 profiler 含逐阶段同步和记录开销，不能与 23.877 ms 的 CUDA Graph 总延迟相加或直接比较；它只用于判断阶段大小。最大的可优化对象是候选 attention/尾部合并，其次是代理检索，而不是 query 准备或 Key append。

一次性成本也不在 Graph token 计时内：同一次 c64 记录中，64K prefill 为 53.98 s，请求局部 QK 因子和索引准备约 0.37 s，ValueSketch 预计算约 0.42 s、安装约 0.002 s。这些成本必须在端到端短回答场景单独报告；多轮或 Agent 复用时才可摊薄。

## 当前 s512 的 Graph 组件分解

Nsight Systems 以 CUDA Graph node 模式记录 4 次 64K replay。Nsight 下 wall time 为 25.280 ms/token，kernel 总和为 24.534 ms/token；干净无 profiler 的正式数字仍是 23.877 ms/token。下表使用 Nsight 的直接测量，不把 profiler 开销隐藏到其他组件。

| 组件 | ms/token | 占 Nsight wall | 具体内容 |
|---|---:|---:|---|
| 模型 dense matmul | 10.870 | 43.00% | QKV/O 投影、MLP GEMM/GEMV |
| 模型其他 kernel | 3.125 | 12.36% | RMSNorm、激活、逐元素运算、cache 写入 |
| 候选 exact attention + Value 尾部合并 | 5.741 | 22.71% | 精确 QK、softmax、V 聚合、ValueSketch 输出合并 |
| 代理扫描、阈值与候选压缩 | 4.554 | 18.01% | 512 样本阈值、全历史低比特扫描、mask compaction |
| Query 投影与量化 | 0.200 | 0.79% | QK-balanced 坐标投影和 INT8 query code |
| suffix 候选追加 | 0.045 | 0.18% | 把 Graph 动态 suffix 作为精确候选加入 |
| Graph launch、同步及未归类间隙 | 0.746 | 2.95% | wall 减去 GPU kernel 总和 |

首先看前四行：标准模型计算占 55.36%，QKSieve 的检索与候选 attention 占 40.72%。Query 投影和 suffix 追加合计不到 1%，继续优化它们没有实际价值。

QKSieve 两个主要阶段进一步分解如下：

| 子阶段 | ms/token |
|---|---:|
| 候选 exact-attention split kernel | 5.576 |
| ValueSketch 最终 tail merge | 0.165 |
| 全历史 threshold-mask + ValueSketch 扫描 | 2.826 |
| 512 样本阈值估计 | 0.822 |
| selection-mask 候选压缩 | 0.706 |
| ValueSketch partial reduction | 0.200 |

这里没有逐 token 的压缩 K/V append：Graph 固定检索 64K prefix，新增 suffix 直接加入 exact 候选，因此只有 0.045 ms 的 suffix ID 追加。Eager 流式路径才需要压缩 Key/Value 增量编码，fast-path CUDA Event 测得约 3.53 ms/token，但它不属于当前 Graph 的 23.877 ms。

从该分解可得一个重要上界：即使不现实地删除全部 QKSieve 检索和 attention kernel，仍剩约 14.74 ms/token 的模型与 runtime。64K Full 的一半是 12.95 ms/token，所以在当前 3090 单 batch 栈上，仅优化 QKSieve kernel 无法达到 2x Full；要超过该界，需要同时优化或量化模型 dense 部分，或者转向更长上下文。

一次性成本在本次复测中为：prefill 53.17 s、ValueSketch 预计算 0.472 s、QK 因子/分配 0.232 s，方法总 setup 约 0.916 s。64K 每 token 只比 Full 省约 2.014 ms；若 setup 每次重建，需要约 455 个生成 token 才能回本。复用所有索引时才从第一个 decode token 开始获得 1.084x 稳态收益。

## 四主题质量

| 变体 | 几何 PPL | 相对 Full PPL 质量 | 平均 top-1 | 平均 KL |
|---|---:|---:|---:|---:|
| Full KV | 6.1475 | 100.00% | 100.00% | 0 |
| 完整 ValueSketch+s512 | 6.1581 | 99.83% | **99.61%** | 待汇总 |
| Unsorted rank-16+s512 | 6.1579 | 99.83% | **99.61%** | 0.000243 |
| NoVS+s512 | 6.0202 | 102.11% | 98.05% | 待汇总 |
| Mean-tail+s512 | 6.2827 | 97.85% | 97.66% | 待汇总 |
| ValueSketch mid 12 | 6.0835 | 101.05% | 98.05% | 0.001895 |
| ValueSketch late 12 | 6.0708 | 101.26% | 98.44% | 0.004315 |
| ValueSketch mid+late 24 | 6.1429 | 100.07% | 98.83% | 0.001130 |

观察：NoVS 的几何 PPL 比 Full 略低，但 top-1 从完整 ValueSketch 的 99.61% 降到 98.05%。这说明尾部补偿并非完全冗余；它主要提高决策稳定性，而不是保证每个短窗口的平均 NLL 更低。

## 失败解释与更新后的假设

- 已通过：阈值采样从约 3,328 降到 512。它使延迟降低约 0.81 ms/token，现有探针未显示系统性质量损失。
- 已否定：用 Mean-tail 代替 ValueSketch。它同时恶化 PPL 和 top-1。
- 已否定：减少 exact attention 分块。操作数看似更少，但 GPU 利用率下降，速度更慢。
- 已否定：删除确定性候选排序。质量不变，但延迟从 23.877 增至 25.235 ms/token；`sortcompact` 是融合压缩路径，不是纯额外排序。
- 尚未通过：完全删除或只在 12 层执行 ValueSketch。四主题 top-1 只有 98.05%–98.44%，低于完整版本 99.61%。
- 已否定：删除前 12 层 ValueSketch。它把 top-1 降到 98.83%。
- 实现受限：rank-12 `sortcompact` 被现有 CUDA kernel 的 rank-16 合同拒绝，尚无质量或速度结论。

更新后的结论：阈值估计可以使用固定 512 样本；大块删除 ValueSketch 层不安全；候选排序和自动 attention 分块都对应更快的融合或并行路径，不能按操作名称直接删除。当前保守版本应冻结为全层 rank-16、sorted ValueSketch+s512。

## 结论边界

目前可以可靠采用的裁剪只有 512 阈值样本，得到 23.877 ms/token 和 1.084x 公平加速。21.411 ms/token 的 NoVS 版本是激进候选，不是可冻结主方法。继续依靠删除外围操作最多只有约 10% 空间，无法产生数量级加速；更大的改进必须针对代理扫描、候选 exact attention 或 ValueSketch kernel 本身。结果只支持 64K、Qwen3-4B、RTX 3090；尚未支持其他长度、模型或完整下游任务。

## 原始结果路径

- 完整 s512：`qksieve_removeops_s512true_64k_20260808`
- NoVS+s512：`qksieve_removeops_novs_s512_auto_64k_20260808`
- Mid+Late 24：`qksieve_removeops_vs_midlate24_graph_64k_20260808` 与四个 `qksieve_vs_midlate24_s512_quality_*` 目录
- Unsorted rank-16：`qksieve_removeops_vs16_unsorted_s512_graph_64k_20260808` 与四个 `qksieve_vs16_unsorted_s512_quality_*` 目录
- Graph node profile：`qksieve_s512_graph_nsys_node_64k_20260808`

上述目录位于远端 `/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807/results/`。
