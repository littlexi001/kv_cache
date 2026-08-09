更新时间：2026-07-17

本文按照“问题、问题分析得到的特性、针对特性设计的方法、实际效果”的顺序，说明当前已经能够真实运行的主方法。当前方法为：

> **RoPE-aware PCA64 INT4 全局 K 索引 + CPU 完整精确 K/V + GPU 精确 K/V 热缓存 + per-query-head 候选检索 + residency-windowed GQA 稀疏 attention。**

该方法不读取答案，不使用 oracle token，不依赖任务标签，不需要针对测试任务训练 router。当前预算仍是固定部署配置与纯长度 gate，不应描述为已经完成的动态预算系统。

## 1. 问题

### 1.1 长上下文的 KV 存储和 attention 读取成本

对包含 `N` 个历史 token、每个 head 维度为 `D` 的自回归模型，标准 decode 每一步都需要读取全部历史 K/V，并计算当前 query 与全部历史 key 的 attention：

\[
S_h=\frac{q_hK_h^T}{\sqrt D},\qquad
O_h=\operatorname{softmax}(S_h)V_h.
\]

因此，KV cache 显存占用和单步 attention 数据读取量都随 `N` 线性增长。128K 上下文下，即使模型权重能够放入 GPU，完整 FP16 K/V 仍会占用大量显存，而且每个 decode step 都要重新扫描全部历史状态。

本项目的目标不是只降低一个逻辑 token ratio，而是同时满足：

1. GPU 持久 KV 状态尽可能接近完整 K/V 的 1%--10%；
2. 质量至少保持 Full KV 的 95%；
3. 在 64K--128K 长上下文中取得可测量的整模型 decode 加速，目标不低于 2.5x；
4. 不删除完整历史信息，未来 query 仍然可以重新访问此前不重要的 token；
5. 不依赖答案、任务标签、oracle 或测试集调参。

### 1.2 只把完整 KV 放到 CPU 不能解决速度问题

如果 GPU 不保存 KV、每一步再从 CPU 读取全部 K/V，显存问题会转化为 PCIe 带宽问题。真正可行的异构方案必须先在 GPU 上用小索引定位少量候选，只搬运当前 query 需要且尚未驻留的精确 K/V。

这意味着系统需要同时解决两个子问题：

- 如何用远小于完整 K/V 的状态准确找到高 attention token；
- 如何避免每一步重复搬运候选 K/V，使稀疏检索的额外开销小于 dense attention 的节省。

### 1.3 GQA 的逻辑需求与物理存储存在冲突

Llama-3.1-8B-Instruct 使用 GQA：多个 query heads 共享一个 KV head。最简单的做法是让同一 GQA 组内的 query heads 共享一组历史 token，但这隐含了一个不成立的假设：共享 K/V 的 query heads 也需要相同证据。

在 128K 开发样本上，PCA32、GQA shared top-2% 的 PPL 质量保持率只有 **78.07%**。增加 shared 候选比例或 PCA 维度仍没有恢复质量，说明问题不只是候选数量不足，而是不同 query heads 需要不同的历史位置。

另一方面，如果为每个 query head 独立选择候选，再把一个 GQA 组内所有候选的并集同时放入 GPU，物理工作集会明显大于单个 head 的逻辑预算。例如每个 query head 选择 1% 时，四个 heads 的候选并集可能超过 3.2%，破坏低显存目标。

所以核心矛盾是：

> **逻辑上必须保留 query-head 独立的证据集合，物理上又不能同时驻留所有逐头候选的并集。**

### 1.4 稀疏 attention 不一定在所有长度上更快

稀疏路径额外包含低维扫描、top-k、cache 查询、PCIe miss fill 和小规模 attention。4K--8K 时，dense attention 本身占整模型 forward 的比例较低，这些固定开销可能大于节省的 attention 时间。因此，一个可部署系统不能强制所有长度都走稀疏路径。

## 2. 问题分析得到的特性

### 2.1 Attention 的“选择信息”比完整 K/V 更容易压缩

最终 attention 输出需要精确 V，但候选定位只需要近似判断哪些 key 与当前 query 更相关。对每层、每个 KV head 的 K 做低秩投影后，64 维近似表示已经能够保留大部分重要 attention mass。

在体育和医学两个真实32K窗口、320个严格配对的 layer-head-query 索引探针中，当前 sampled PCA64 INT4 对精确 top-2% token 的平均集合召回率为 **68.11%**，但平均 oracle attention mass recall 达到 **98.27%**，p10仍为 **95.81%**。两者的差异说明：低维索引不必逐个复现完整 top-k 排名，只要优先保住承载主要 attention mass 的 token，就可以作为候选定位器。

由此得到第一个设计原则：

> **压缩 K 的选择表征，保留被选中 K/V 的原始精度；低维索引只负责检索，不直接替代最终 attention。**

### 2.2 不同 query heads 的证据具有互补性

同一 GQA 组内的 query heads 虽然共享原始 K/V，但 query 投影不同、功能不同，产生的高分历史位置也不同。128K 下 shared 候选失败，而 PCA64 INT4、per-query-head 直接 top-1% 将质量保持恢复到 **96.34%**，接近 exact-QK top-1% 的结果。

由此得到第二个设计原则：

> **候选必须在 query-head 粒度生成，不能先平均多个 query heads 的分数再选一组共享 token。**

### 2.3 逻辑候选与物理驻留可以解耦

每个 query head 需要独立候选，不代表所有 heads 的候选必须同时常驻 GPU。attention 输出本来就按 head 独立计算，因此可以把同一 GQA 组拆成较小窗口：先为一个或两个 query heads 填充候选并计算 attention，再复用同一批物理 cache slots 处理后续 heads。

由此得到第三个设计原则：

> **逐头候选是逻辑视图，GPU exact cache 是有界物理工作集；通过流式执行把二者解耦。**

### 2.4 相邻 decode steps 的候选存在时间局部性

虽然重要 token 会随 query 改变，但相邻生成位置的候选并非完全随机。128K speed-first 实验中，3.2% 精确 K/V 热缓存对逐头候选的平均命中率为 **81.52%**。因此没有必要每一步都重新从 CPU 搬运全部 top-k，可以用持久 hot cache 复用近期访问过的精确 K/V。

由此得到第四个设计原则：

> **完整历史 K/V 保存在 CPU，GPU 使用物理有界的精确热缓存，只填充当前候选中的 miss。**

### 2.5 稀疏路径的收益随上下文长度增长

PCA64 扫描和目录查询的固定成本远小于完整 FP16 K/V 的增长速度。在约 7.5K 的 LongBench 样本上，当前 raw sparse 路径仍慢于 Full；在 128K 上，dense attention 的读取成本已经足够大，当前实现能够取得约 2.7x 的整模型 decode 加速。

由此得到第五个设计原则：

> **按纯上下文长度选择 Full 或 sparse；短上下文走 Full，长上下文才启用层次化检索。**

## 3. 针对特性设计的方法

### 3.1 三层状态布局

每层、每个 KV head 的历史状态分为三层：

1. **CPU pinned memory：完整 FP16 K/V。** 所有历史 token 的精确 K/V 都保留，任何未来 query 都可以重新取回，不存在不可逆 eviction。
2. **GPU：PCA64 INT4 全局 K 索引。** 每个历史 key 保存一个64维、4-bit 的近似代码和一个 FP16 scale，用于全局候选扫描；V 不进入全局索引。
3. **GPU：精确 FP16 K/V hot cache。** 默认容量约为完整 K/V 的3.2%，由 fused hash directory 和 LRU replacement 管理。

索引负责回答“可能需要哪些 token”，CPU 完整状态负责保证可恢复性，GPU 精确热缓存负责降低反复搬运成本。

### 3.2 RoPE-aware PCA64 INT4 索引

模型先用标准 forward 产生 post-RoPE key。对某层、某 KV head：

\[
K=[k_1,\ldots,k_N]^T\in\mathbb{R}^{N\times128}.
\]

当前实现每隔32个 token 采样一次，构造未中心化二阶矩：

\[
C=\frac{1}{|\mathcal S|}\sum_{i\in\mathcal S} k_i^Tk_i.
\]

取最大的64个特征向量组成投影矩阵：

\[
U\in\mathbb{R}^{128\times64},\qquad z_i=k_iU.
\]

对每个 token 的 `z_i` 独立做对称 INT4 量化：

\[
s_i=\frac{\max_j|z_{ij}|}{7},\qquad
c_{ij}=\operatorname{clip}\left(\operatorname{round}\frac{z_{ij}}{s_i},-7,7\right).
\]

64个 INT4 code 加一个 FP16 scale 相对于完整 FP16 K+V 的存储比例为：

\[
R_{\mathrm{index}}
=\frac{64\times4+16}{2\times128\times16}
=6.640625\%.
\]

索引直接建立在 post-RoPE K 上，query 也使用标准 RoPE 后的 Q。方法不重写 position id，不把稀疏 token 重新编码成虚假连续位置。

### 3.3 Per-query-head 在线候选检索

对当前 query head `h` 的 post-RoPE query `q_h`：

1. 计算低维 query `q_hU`；
2. CUDA INT4 kernel 扫描对应 KV head 的全部历史索引；
3. 该 query head 独立选择近似分数最高的 `ceil(rho*N)` 个历史 token；
4. 当前 token 始终以精确 K/V 加入候选；
5. 主结果不额外强制加入固定 recent window。

当前报告配置中，候选比例就是最终 attention 比例，没有先扩大候选再做 exact rerank：

| 场景 | 每个 query head 的候选比例 | Exact hot cache |
|---|---:|---:|
| LongBench 完整质量实验 | 2.5% | 3.2% |
| 128K speed-first | 1.0% | 3.2% |
| 128K quality-first | 1.5% | 4.1% |

### 3.4 Cache hit/miss 与精确稀疏 attention

得到候选位置后，系统在 fused hash directory 中查询其精确 K/V 是否驻留 GPU：

- cache hit：直接读取 GPU slot；
- cache miss：从 CPU pinned memory 把该 token 的完整 FP16 K/V 填入 GPU，并更新 LRU；
- 候选到齐后，使用完整维度 K/V 计算精确稀疏 attention：

\[
O_h=\operatorname{softmax}
\left(\frac{q_hK_{S_h}^{T}}{\sqrt{128}}\right)V_{S_h}.
\]

因此，PCA 和 INT4 的误差只影响“选中了谁”，不会继续进入最终 value aggregation。

### 3.5 Residency-windowed GQA

系统先为每个 query head 生成独立候选集合，但不同时物化所有集合。当前128K配置在一个 GQA 组内按两个 query heads 为一批执行：

```text
for each layer:
    candidate_ids = PCA64_INT4_scan(all_query_heads)
    for each two-query-head window in a GQA group:
        query exact-cache directory
        fill cache misses from pinned CPU K/V
        run exact sparse attention for this head window
        reuse the same physical slots for the next window
```

这种执行方式保留了 query-head 独立性，同时让物理 exact cache 容量由固定 hot-cache budget 决定，而不是由四个 query heads 的候选并集决定。

### 3.6 Prompt 协议与长度 gate

面向生成任务，当前协议为 `full_prompt_then_compress`：

1. 完整 prompt 使用标准 dense prefill；
2. 保存 dense prefill 产生的首个 answer logits；
3. 将 prompt KV 转换为层次化状态；
4. 后续生成 token 使用层次化稀疏 attention。

当前部署策略只读取 prompt token 数：

```text
prompt length < 16K:  Full KV
prompt length >= 16K: hierarchical sparse KV
```

这个 gate 不读取任务类型、答案或测试分数。它解决短上下文固定检索开销无法摊销的问题，但不是动态质量 router。

## 4. 效果

### 4.1 总体结果

当前已完成实验的核心结果如下：

| 场景 | Full | 当前方法 | 质量保持率 | GPU KV ratio | Decode speedup | Protocol/E2E speedup |
|---|---:|---:|---:|---:|---:|---:|
| LongBench 16任务，3750样本 | 0.376365 | 0.363799 | **96.66%** | **10.62%** | 旧协议不加速 | 旧协议不加速 |
| LongBench 新协议，160样本 | 0.371040 | 0.365805 | **98.59%** | **10.66%** | 0.283x raw sparse | 长度 gate 后等于 Full |
| 128K religion 单窗口 PPL | 15.1605 | 15.7358 | **96.34%** | **9.99%** | **2.706x** | **1.126x** |
| 128K computer 单窗口 PPL | 60.4325 | 60.5298 | **99.84%** | **11.00%** | **2.383x** | **1.069x** |

这里的 GPU KV ratio 是持久 GPU cache tensor bytes 相对于完整 FP16 K/V bytes 的比例，不是单步 attention token ratio。约10%的持久状态主要由6.64%的 PCA64 INT4 全局 K 索引、3.2%--4.1%的精确热缓存以及 basis、directory 等元数据组成；每个 query head 真正参加 attention 的历史 token 只有1%--2.5%。

### 4.2 LongBench 质量

完整 LongBench 实验覆盖16个英文任务、3750个样本，每个样本运行 Full 和 sparse，共7500条配对结果：

\[
\text{Full Macro}=0.376365,qquad
\text{Sparse Macro}=0.363799.
\]

当前方法在10.62% GPU KV下保持 Full 的96.66%。16个任务中有11个达到至少95%的 Full-relative 质量。主要弱项是 Musique、PassageCount 和 Qasper，三者贡献约87.4%的总体 Macro gap；排除它们仅用于误差分析时，其余13个任务保持99.51%，但论文主结果必须继续报告完整16任务。

这批3750样本使用较早的 `prefix_sparse_suffix` 协议，逐 token 重放 question suffix，因此只能作为完整质量表，不能作为速度结果。新 `full_prompt_then_compress` 协议目前完成了每任务10条、共160条验证，质量保持率提升到98.59%，但样本平均上下文只有约7.5K，raw sparse online speed 仍只有0.283x。16K长度 gate 会让这些短样本走 Full。

### 4.3 128K 质量、显存与速度闭环

128K speed-first 配置使用 per-query-head 1%、3.2% exact hot cache 和 stream group 2：

| 指标 | Full KV | 当前方法 |
|---|---:|---:|
| PPL | 15.1605 | 15.7358 |
| 质量保持率 | 100.00% | **96.34%** |
| Persistent GPU KV ratio | 100.00% | **9.99%** |
| Dense prefill | 215.069 s | 215.552 s |
| Cache conversion | 0 s | 24.213 s |
| Online forward | 93.809 s | 34.663 s |
| Protocol total | 308.878 s | 274.429 s |
| Decode speedup | 1.000x | **2.706x** |
| Protocol speedup | 1.000x | **1.126x** |
| Exact-cache hit rate | -- | **81.52%** |

2.706x 是包含 PCA64 INT4 扫描、top-k、directory、PCIe miss fill、精确稀疏 attention、MLP、投影和其他模型计算的整模型 decode speedup，不是 attention FLOPs 理论上界。

128K quality-first 配置把候选比例提高到1.5%、exact cache提高到4.1%，在独立 computer 窗口上得到99.84%质量保持、11.00% GPU KV和2.383x decode speedup。两个窗口来自不同主题，不能直接据此推断预算变化的因果收益，但它们给出了当前已经真实运行的两个 Pareto 工作点。

### 4.4 在线速度瓶颈

128K、per-query-head 1% 配置下，独立稀疏 attention 子系统的每层时间为：

| 组件 | 每层时间 | 占稀疏子系统 |
|---|---:|---:|
| PCA64 INT4 scan + top-k | 0.212 ms | 21.1% |
| Fused hash/LRU directory | 0.050 ms | 5.0% |
| PCIe miss fill + sparse attention | 0.741 ms | 73.9% |
| **合计** | **1.003 ms** | **100%** |

当前瓶颈不是 hash directory，而是 cache miss 的精确 K/V 搬运与稀疏 attention。按完整 forward 估算，非 attention 的 MLP、投影、LayerNorm 和运行时成本约占 sparse decode 的73.7%；即使检索和 attention 时间降到零，当前模型和软件栈的 decode speedup 上限也约为3.65x。

### 4.5 当前结论边界

当前结果支持的结论是：

> 在无需 oracle、任务标签或训练 router 的条件下，PCA64 INT4 全局 K 索引、per-query-head 候选和 residency-windowed GQA 可以把持久 GPU KV 降到约10%，让每个 query head 只对1%--2.5%的历史 token 做精确 attention；该方法在完整 LongBench 上保持96.66% Macro，并在一个128K开发窗口上取得2.706x整模型 decode 加速。

当前结果还不能支持以下更强主张：

- 128K 的2.706x已经在多主题、多窗口和长生成上稳定复现；
- 固定1%预算对所有文本都能保持至少95%质量；
- LongBench 的完整3750样本已经证明端到端加速；
- 64K/128K RULER、Long ICL和多模型实验已经完成；
- 当前静态预算已经是通用动态 router。

因此，当前方法已经形成完整的问题诊断、机制设计和物理运行闭环，但论文级结论仍需要冻结配置后完成独立多主题128K验证、RULER/Long ICL、多模型以及长生成摊销实验。
