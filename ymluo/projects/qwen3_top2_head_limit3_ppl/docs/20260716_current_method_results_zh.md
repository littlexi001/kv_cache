# 当前层次化 KV 检索方法与实验结果

更新时间：2026-07-16

## 1. 当前结论

当前可实际运行的主方法是 **RoPE-aware PCA64 INT4 全局 K 索引 + CPU 完整 K/V + GPU 精确 K/V 热缓存 + per-query-head 稀疏 attention**。方法不读取答案、不使用 oracle token、不依赖任务标签，也不需要为当前模型训练 router。

目前最重要的已验证结果为：

| 场景 | Full | 当前方法 | 质量保持率 | GPU KV ratio | Decode speedup | Protocol/E2E speedup |
|---|---:|---:|---:|---:|---:|---:|
| LongBench 16 任务，3750 样本 | 0.376365 | 0.363799 | **96.66%** | **10.62%** | 旧协议不加速 | 旧协议不加速 |
| LongBench 更新协议，160 样本 | 0.371040 | 0.365805 | **98.59%** | **10.66%** | 0.283x | 长度 gate 后等于 Full |
| 128K religion 单窗口 PPL | 15.1605 | 15.7358 | **96.34%** | **9.99%** | **2.706x** | **1.126x** |

其中 128K 的 2.706x 是生成阶段的整模型 decode speedup，Full 基线使用全部 128K K/V 做标准 dense attention。该数字包含 PCA64 INT4 扫描、top-k、GPU cache 查询、PCIe miss fill、精确稀疏 attention、MLP 和其他模型计算；它不是 attention 理论上界。

## 2. 方法

### 2.1 系统状态

对每一层和每个 KV head，完整历史状态分为三部分：

1. **CPU pinned memory 中的完整 FP16 K/V**：保存所有历史 token 的精确 K/V，作为最终 attention 和 cache miss 的数据源。
2. **GPU 中的 PCA64 INT4 全局 K 索引**：为每个历史 K 保存 64 维、4-bit 的近似表征及 FP16 scale，用于扫描和候选定位；V 不进入全局索引。
3. **GPU 中的精确 FP16 K/V hot cache**：默认容量为完整 K/V 的 3.2%，使用 fused hash directory 和 LRU replacement 管理。

当前索引直接基于模型标准 forward 产生的 post-RoPE K 构建，query 也使用同一位置编码后的 Q。方法不修改 position id，不重新训练 RoPE，也不对保留 token 分配虚假连续位置。

### 2.2 PCA64 INT4 索引

设某层、某 KV head 的 post-RoPE key 为

\[
K=[k_1,\ldots,k_N]^T\in\mathbb{R}^{N\times 128}.
\]

当前实现每隔 32 个 token 采样一次，构造未中心化二阶矩：

\[
C=\frac{1}{|\mathcal{S}|}\sum_{i\in\mathcal{S}}k_i^T k_i.
\]

取最大 64 个特征向量组成投影矩阵：

\[
U\in\mathbb{R}^{128\times64},\qquad z_i=k_iU.
\]

每个 token 独立进行对称 INT4 量化：

\[
s_i=\frac{\max_j|z_{ij}|}{7},\qquad
c_{ij}=\operatorname{clip}\left(\operatorname{round}\frac{z_{ij}}{s_i},-7,7\right).
\]

64 个 INT4 code 占 256 bit，一个 FP16 scale 占 16 bit。相对于一个 token 的完整 FP16 K+V：

\[
R_{\mathrm{index}}
=\frac{64\times4+16}{2\times128\times16}
=6.640625\%.
\]

### 2.3 在线候选检索

对当前 query head 的 post-RoPE query `q`：

1. 计算低维 query `qU`；
2. 使用 CUDA INT4 kernel 扫描该 KV head 的全部历史索引；
3. 每个 query head 独立选择近似分数最高的 `ceil(rho*N)` 个历史 token；
4. 在 fused hash directory 中查询候选的精确 K/V 是否已驻留 GPU；
5. cache miss 从 CPU pinned memory 搬入 GPU，并更新 LRU；
6. 对候选精确 K/V 和当前 token 执行完整维度 attention：

\[
O_h=\operatorname{softmax}
\left(\frac{q_hK_{S_h}^{T}}{\sqrt{128}}\right)V_{S_h}.
\]

当前已报告配置中 `candidate_fraction = attention_fraction`，因此近似索引选出的候选直接进入最终 attention，没有先扩大候选再 exact rerank。当前 token始终以精确 K/V 参加 attention；主结果没有额外强制保留固定 recent window。

### 2.4 Per-head stream

Llama-3.1-8B-Instruct 使用 GQA。当前 `per_head_stream` 为每个 query head 保留独立候选集合，再按 1、2 或 4 个 GQA group 分批获取精确 K/V 和计算 attention，避免把所有 query head 的候选并集同时常驻 GPU。

已报告配置为：

| 场景 | 候选/attention 比例 | Exact hot cache | Stream group |
|---|---:|---:|---:|
| LongBench 3750 样本 | 2.5% | 3.2% | 1 |
| 128K speed-first | 1.0% | 3.2% | 2 |
| 128K quality-first | 1.5% | 4.1% | 2 |

预算目前是部署配置，不是 oracle，也不是已冻结的动态 router 输出。

### 2.5 Prompt 与长度策略

当前面向生成任务的集成协议为 `full_prompt_then_compress`：

1. 完整 prompt 使用标准 dense prefill；
2. 保存 dense prefill 产生的首个 answer logits；
3. 将完整 prompt KV 转换为层次化状态；
4. 后续生成 token 使用稀疏检索 attention。

该协议保证问题和完整上下文都已进入模型状态，不需要在压缩后逐 token 重放 question suffix。

当前实用长度 gate 为：prompt 小于 16K 时使用 Full KV，16K 及以上才启用层次化 cache。该 gate 只读取 token 长度，不读取任务类型、答案或测试分数。

## 3. LongBench 结果

### 3.1 完整 16 任务主表

模型为 Llama-3.1-8B-Instruct。完整实验包含 LongBench 16 个英文任务、3750 个样本；每个样本分别运行 Full KV 和当前稀疏 cache，共 7500 条配对结果。

| 任务 | Full KV | Sparse | 质量保持率 | GPU KV ratio |
|---|---:|---:|---:|---:|
| 2WikiMQA | 0.4677 | 0.4690 | 100.27% | 10.67% |
| GovReport | 0.2106 | 0.2058 | 97.73% | 10.39% |
| HotpotQA | 0.4855 | 0.4690 | 96.59% | 10.48% |
| LCC | 0.6321 | 0.6832 | 108.08% | 11.36% |
| MultiNews | 0.1601 | 0.1586 | 99.04% | 11.40% |
| MultiFieldQA-en | 0.5596 | 0.5183 | 92.62% | 10.76% |
| Musique | 0.2814 | 0.1967 | 69.91% | 10.44% |
| NarrativeQA | 0.2378 | 0.2413 | 101.48% | 10.52% |
| PassageCount | 0.0988 | 0.0648 | 65.51% | 10.41% |
| PassageRetrieval-en | 0.7250 | 0.7200 | 99.31% | 10.35% |
| Qasper | 0.4541 | 0.3972 | 87.47% | 10.88% |
| QMSum | 0.1726 | 0.1707 | 98.91% | 10.80% |
| RepoBench-P | 0.5127 | 0.5502 | 107.32% | 10.05% |
| Samsum | 0.1432 | 0.1476 | 103.08% | 10.45% |
| TREC | 0.7000 | 0.6550 | 93.57% | 10.62% |
| TriviaQA | 0.1807 | 0.1735 | 96.03% | 10.19% |
| **Macro** | **0.376365** | **0.363799** | **96.66%** | **10.62%** |

16 个任务中有 11 个达到至少 95% 的 Full-relative 质量。主要弱项为 Musique、PassageCount 和 Qasper；三者贡献约 87.4% 的总体 Macro gap。其余 13 个任务的 Full/Sparse Macro 为 0.39904/0.39709，即 99.51% 质量保持率。该 13 任务数字只用于定位误差来源，论文主结果仍必须使用全部 16 个任务。

### 3.2 协议说明

上述 3750 样本主表使用较早的 `prefix_sparse_suffix` 协议：先转换 prefix，再逐 token 重放 question suffix。它提供完整质量主表，但会产生很大的 replay 开销，不能作为当前系统的速度结果。

更新后的 `full_prompt_then_compress` 已在每任务 10 条、共 160 条样本上验证：

| 方法 | Macro score | 质量保持率 | GPU KV ratio | Raw online speed |
|---|---:|---:|---:|---:|
| Full KV | 0.371040 | 100.00% | 100.00% | 1.000x |
| Sparse | 0.365805 | **98.59%** | **10.66%** | 0.283x |

这些样本平均上下文约 7.5K，稀疏路径的固定检索开销大于 dense attention 节省。因此当前长度 gate 在这些样本上选择 Full KV。

## 4. 128K 质量与速度

### 4.1 Speed-first 配置

配置：religion window 0、128000 history token、256 query token、32 个评测 token、PCA64 INT4、每 query head 1%、3.2% exact cache、stream group 2。

| 指标 | Full KV | Sparse |
|---|---:|---:|
| PPL | 15.1605 | 15.7358 |
| PPL-relative 质量保持率 | 100.00% | **96.34%** |
| Persistent GPU KV ratio | 100.00% | **9.99%** |
| Dense prefill | 215.069 s | 215.552 s |
| Cache conversion | 0 s | 24.213 s |
| Online forward | 93.809 s | 34.663 s |
| Protocol total | 308.878 s | 274.429 s |
| Decode speedup | 1.000x | **2.706x** |
| Protocol speedup | 1.000x | **1.126x** |
| Mean exact-cache hit rate | -- | **81.52%** |

该实验中的 online forward 包含 256 个 query replay step 和随后 31 个 target forward。它证明在 128K、足够多在线 step 的条件下，当前物理实现能够相对 full dense attention 获得约 2.7x 的整模型 decode 加速；它尚不能替代 `full_prompt_then_compress` 下多主题、长生成的独立 E2E 结果。

### 4.2 Quality-first 配置

在 computer 128K 单窗口上，每 query head 1.5%、4.1% exact cache、stream group 2 的结果为：

| Full PPL | Sparse PPL | 质量保持率 | GPU KV ratio | Decode speedup | Protocol speedup |
|---:|---:|---:|---:|---:|---:|
| 60.4325 | 60.5298 | **99.84%** | **11.00%** | **2.383x** | **1.069x** |

两个 128K 点来自不同主题，不能据此单独估计预算变化的因果收益；它们分别表示当前已验证的 speed-first 和 quality-first 工作点。

## 5. 当前速度瓶颈

128K、每 query head 1% 配置下，独立稀疏 attention 子系统的每层计时为：

| 组件 | 每层时间 | 占稀疏子系统 |
|---|---:|---:|
| PCA64 INT4 scan + top-k | 0.212 ms | 21.1% |
| Fused hash/LRU directory | 0.050 ms | 5.0% |
| PCIe miss fill + sparse attention | 0.741 ms | 73.9% |
| **合计** | **1.003 ms** | **100%** |

主要瓶颈是未命中候选的精确 K/V 搬运与稀疏 attention，不是 hash directory。把每层结果近似映射到 32 层完整 forward：

| 组成 | 每次 Sparse forward | 占 Sparse decode |
|---|---:|---:|
| PCA64 scan + top-k | 6.8 ms | 5.6% |
| Directory | 1.6 ms | 1.3% |
| PCIe fill + sparse attention | 23.7 ms | 19.4% |
| MLP、投影、LayerNorm、通信和其他运行时成本 | 约 89.9 ms | 73.7% |
| **合计** | **约 122.0 ms** | **100%** |

最后一行中的非 attention 成本由完整 forward 减去独立子系统时间估算，不是同一次逐 kernel profiler 的直接测量。按该估算，即使把稀疏检索与 attention 时间降到零，当前模型和软件栈的 decode speedup 上限也约为 3.65x。

## 6. 当前结果的适用范围

- LongBench 3750 样本结果已经完整，但使用旧的 question replay 协议；更新协议目前只有 160 样本验证。
- 128K 2.706x 是单主题、单窗口、32 个 target token 的开发结果，不是多主题论文主结果。
- 当前没有可用于论文主张的 64K/128K RULER 最终汇总；相关实验仍在运行。
- 当前预算是固定工作点和纯长度 gate，不应描述为已经完成的通用动态 router。
- 当前可靠主张是：约 10% 持久 GPU KV、1%--2.5% per-head 精确 attention，可以在 LongBench 保持 96.66% Macro，并在一个 128K 开发窗口达到约 2.7x decode speedup。

## 7. 结果位置

LongBench 16 任务完整结果：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  outputs/20260716_hierarchical_longbench_full_v1_merged/
```

128K speed Pareto：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  results/20260716_128k_speed_pareto/
  results/20260716_128k_speed_pareto_summary/
```

主要实现：

```text
src/hierarchical_pca_cache_20260715.py
src/hierarchical_cache_cuda_20260715.py
src/run_hierarchical_longbench_probe_20260715.py
src/run_hierarchical_ruler_probe_20260716.py
src/run_hierarchical_physical_cache_ppl_20260715.py
```
