# QKSieve：当前方法、证据与边界

日期：2026-08-07  
状态：数值方法冻结；端到端执行优化进行中。

## 一句话结论

QKSieve 的目标是在不丢弃原始 KV 的前提下，用低比特索引定位每个 query head 的少量重要历史 token；候选使用原始 FP16 K/V 精确计算，未选 token 用 ValueSketch 补偿。已有长上下文质量证据很好，但使用公平 Native-GQA Full 基线后，32K/64K 的整模型 Decode 仍未加速。当前主要瓶颈是 kernel launch 与 Python/HF 调度，不是稀疏 Attention 的 GPU 计算量。

## 1. 问题与已知现象

Full Attention 对长度为 `N` 的历史逐 token 读取全部 K/V，因此 Decode 的 Attention 成本随 `N` 线性增长。

在多组 held-out 诊断中，每层每个 query head 只需保留少量高权重历史位置，便能接近 Full Attention。QKSieve 不把这个结论简化成“永久只存 2% KV”：不同长度下候选预算连续变化，以避免短文本候选过少、长文本候选无限增长。

当前候选预算为：`B(N) = min(N, max(256, min(ceil(0.06N), 1280)))`。

这意味着每个 query head 最多精确处理 1,280 个 token：2K 时至少 256 个，64K 时约 2%，128K 时约 1%。

## 2. 冻结的数值方法

配置名：`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`。

1. **请求内 QK-balanced 坐标**：对每层、每个 KV head 的 Query/Key 二阶矩做变换。完整 128 维变换保持 `q'^T k' = q^T k`，因此坐标变换本身不引入 QK 误差；它只把联合 QK 能量排序到前面的 band。
2. **8 个 16 维 band 的自动位宽**：每个 band 从 `0/1/2/4/8 bit` 中选位宽。qMSE + OAS 根据该 head 的 Query 分布分配位宽，总 Key 编码预算不超过 240 bit/token/KV-head。
3. **确定性 sampled-quantile 检索**：按目标候选比例自适应采样，保证样本中约有 64 个上尾锚点；随后一次低比特全历史扫描直接产生候选 bitmask，不使用通用排序，也不做 exact-QK rerank。
4. **候选精确 Attention**：只对候选读取 GPU 常驻的原始 FP16 K/V，计算精确 QK、softmax 与 Value 分子。
5. **未选 token 的 ValueSketch 补偿**：未选位置以 `W_O`-aware、rank-16、block-256、INT4 的 ValueSketch 累计近似 softmax 分子/分母，再与候选精确结果在同一 softmax 标尺下合并。

QKSieve 是 query-aware sparse Attention，不是 KV eviction：原始 K/V 仍可寻址，索引只降低每一步实际扫描和精确计算的规模。

## 3. 存储与质量证据

| 项目 | 当前数值 |
|---|---:|
| Key 低比特索引 | 完整 FP16 K+V 的约 5.86% |
| ValueSketch | 约 1.61% |
| 辅助索引合计 | 约 7.47% |
| 16 个英文 LongBench 任务，3,750 严格配对 | 参考路径保持 Full 的 99.881% |
| LongBench macro：Full / QKSieve reference | 0.459398 / 0.458852 |
| paired-bootstrap 95% CI（保持率） | [99.424%, 100.347%] |

LongBench 结果来自 Llama-3.1-8B-Instruct 的完整 proxy/top-k 质量参考路径。它证明低比特 QK 表征可以保持下游质量；但它**不是**当前 sampled-quantile 部署路径与现行内核的联合速度证明。论文最终必须在同一冻结配置上重新补齐质量和速度。

## 4. 公平测速后的真实状态

旧文档中的 `1.9x–3.1x` 整模型 Decode 加速不再作为正式结论：旧 Full 基线物化了 GQA 的重复 K/V，慢于真实 Native-GQA SDPA。

在 Qwen3-4B-Instruct-2507、RTX 3090、同一教师强制 Decode 测试中，旧的
32K/64K Native-GQA Full 仍包含 strided K/V 的 `.contiguous()` 复制，因此下表
仅作为历史记录：

| 历史长度 | Full | QKSieve | QKSieve / Full |
|---:|---:|---:|---:|
| 32K | 43.90 ms/token | 48.06 ms/token | 0.91x |
| 64K | 52.43 ms/token | 63.49 ms/token | 0.83x |

2026-08-08 的复核发现：第一版 Native-GQA Full 仍对预分配 cache 的 strided
K/V 执行 `.contiguous()`。删除该完整 K/V copy，并在确认 decode mask 全零后
直接使用 `enable_gqa=True`，得到以下严格配对：

| 历史长度 | GPU | 零复制 Full | QKSieve 稳态 | Full / QKSieve |
|---:|---:|---:|---:|---:|
| 64K | 1 x RTX 3090 | 42.63 ms/token | 67.85 ms/token | 0.628x |
| 128K | 2 x RTX 3090 | 61.23 ms/token | 71.29 ms/token | 0.859x |

64K CUDA Graph 也按相同零复制 Full 口径复核：Full fixed-position Graph 为
25.891 ms/token，QKSieve 真实增长 Graph 为 24.689 ms/token，公平加速为
**1.049x**。旧 Graph 表中的 2.064x，以及 Legacy HF Full 得到的 3.40x，均
正式作废。

这说明当前普通执行不存在“多生成一些 token 就一定回本”的情况：即使索引成本
为零，在线 Decode 仍较慢。CUDA Graph 可以消除大部分 launch gap，但相对强
Full Graph 的优势目前只有约 5%。

但 64K 的 Nsight kernel 统计同时显示：

| 指标 | Native-GQA Full | QKSieve |
|---|---:|---:|
| GPU kernel 总工作量 | 49.69 ms/token | 24.83 ms/token |
| 相对 GPU 工作量 | 1.00x | 约 2.00x 更少 |

因此问题不是 QKSieve 的数值计算更重，而是当前实现约有 2,366 次 kernel launch/token，且存在 Python、HF cache/control-flow 与小 kernel 间的空隙。GPU 已经完成了应有的约 2 倍计算缩减，但墙钟时间尚未兑现。

## 5. 正确的性能报告方式

质量与速度必须使用相同的数值方法、相同的候选预算和相同的补偿公式；但不能只报告一种请求生命周期。

| 指标 | 包含内容 | 适用场景 |
|---|---|---|
| Quality | 完整冻结方法的 NLL/任务分数 | LongBench、RULER、PPL |
| Cold request | Prefill + 建索引 + 全部 Decode | 单轮普通请求 |
| Warm cached-prefix | 已有 KV/索引后的新问题与 Decode | 多轮对话、Agent |
| Steady Decode | 不含一次性建索引的逐 token 延迟 | 稀疏 Attention 的在线效率 |
| Attention subsystem | 检索、补偿、稀疏 Attention | 定位内核优化收益 |

短 LongBench 输出无法充分均摊索引成本，这不是测试错误，而是该场景的真实限制。对于 Agent 的可复用长前缀，必须额外证明索引或坐标是否可以跨请求复用；当前 request-local 坐标尚不能直接假定可复用。

## 6. 下一步与论文边界

下一步只做不改变候选或数值公式的执行优化：

1. 固定前缀的 CUDA Graph，消除每 token 大量 Python 与 kernel launch 开销。
2. 融合 Key/Value 增量编码、Query 投影、阈值扫描和候选消费，减少每层小 kernel。
3. 验证“静态稀疏前缀 + 精确动态后缀”：前缀索引一次构建，新增问题/生成 token 始终精确计算，并严格检查候选 ID、NLL 和 Top-1 等价性。

论文当前可以主张：低比特 QK 索引与 ValueSketch 在长上下文质量上具备强证据，且 Attention GPU 工作量显著下降。

论文当前不能主张：QKSieve 已在公平 Native-GQA 基线下实现 32K/64K 的整模型端到端加速，或 LongBench 的 99.881% 与某个旧速度数字来自同一部署路径。

## 7. 复现入口

- 主方法：`src/run_head_top2_targeted_ppl_20260714.py`
- 低比特检索：`src/mixedblock_spectral_cuda_20260729.py`
- ValueSketch 与候选/尾部合并：`src/qksieve_valuesketch_cuda_20260801.py`
- LongBench 严格配对结果：`results/20260728_qksieve_three_model_longbench/llama31_8b/paired_summary.json`
