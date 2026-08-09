# Per-Head Hierarchical PCA KV Cache：面向 128K 上下文的逐头证据检索与分层物理驻留

> **一句话结论：** 在 Llama-3.1-8B-Instruct、128K 上下文和真实 HuggingFace cache 生命周期中，当前主方法使用 PCA64 INT4 压缩索引、per-query-head top-1% 虚拟候选和双-head residency-windowed GQA，在 **9.9918% GPU 常驻 KV 状态**下取得 **96.34% PPL 质量保持**和 **2.686x decode 加速**；包含 prefill、cache 转换和 decode 后，总流程加速为 **1.143x**。该结果已经验证核心机制，但目前只覆盖一个 128K 主题窗口的前 32 个目标 token，尚不能写成跨任务、跨模型的普适结论。

**作者：** Yiming Luo，Fudan University  
**日期：** 2026-07-15  
**文档状态：** 内部技术报告，方法与核心实现已完成，外部验证仍在进行  
**工作名称：** Per-Head Hierarchical PCA KV Cache

## 摘要

长上下文推理的 KV cache 同时面临显存占用和 attention 计算量随上下文长度线性增长的问题。已有 token eviction、固定 block/page 保留和 GQA 共享检索通常隐含一个假设：同一层或同一 KV head 服务的多个 query heads 可以共享相同的历史 token 集合。我们的诊断表明，这个假设在 128K 上下文下会成为主要质量瓶颈。不同 query heads 会检索互补的证据；把它们强制合并为一个共享候选集会遗漏 head-specific evidence，而同时常驻所有逐头候选又会造成 GQA 候选并集膨胀，破坏低显存目标。

本报告提出一个三层结构。第一层使用每层、每个 KV head 的 PCA 子空间和 INT4 量化代码构建低成本全局索引；第二层为每个 query head 独立产生 top-k 虚拟候选，使检索粒度与真实 attention 需求一致；第三层不同时驻留全部逐头候选，而是在同一 GQA 组内按两个 query heads 为一个窗口，依次把候选 K/V 从 pinned CPU memory 填入 3.2% GPU 热缓存并计算 ragged attention。完整精确 K/V 保存在 CPU，GPU 只常驻 PCA 索引、精确热缓存和 hash/LRU 目录。

当前主结果在 128K 上首次同时满足三个预设目标：低于 10% 的物理常驻 KV、超过 95% 的质量保持和超过 2.5x 的真实 decode 加速。实验也明确暴露了边界：固定 1% 在独立 computer 主题上只有 91.87% 质量保持，提升到 1.5% 可恢复到 99.84%，但速度降至 2.278x；4K--8K 短上下文下固定稀疏执行不具备速度优势。因此，最终系统需要长度感知和位置风险感知的预算控制，而不能把 1% 写成全局常数。

## 0. 研究状态与结论边界

### 0.1 已经得到支持的命题

| 命题 | 当前判定 | 主要证据 |
|---|---|---|
| PCA 低维索引可以低成本召回高 attention token | 支持 | PCA64 INT4 索引约占 Full KV 的 6.64%，candidate recall@top2 为 0.9588 |
| GQA 组内 query heads 需要不同的历史证据 | 支持 | shared 候选在 128K 明显退化；per-head 虚拟候选把质量保持恢复到 96.34% |
| 逐头候选必须全部同时驻留 GPU | 否定 | 双-head 分窗口复用同一 3.2% 热缓存，质量不变且常驻比例降至 9.9918% |
| 128K 下低于 10% KV 可以获得超过 2.5x decode 加速 | 支持 | 真实模型逐 token 同步计时为 2.686x |
| 当前固定 1% 动作可以跨主题直接使用 | 否定 | religion 为 96.34%，独立 computer 只有 91.87% |
| 当前结果已经证明跨任务、跨模型通用性 | 证据不足 | 仍缺更多独立窗口、任务型生成和第二模型 |

### 0.2 当前最可靠的主张

当前结果应表述为：

> 对真正长上下文，逐 query head 的证据集合与物理 GQA 共享结构之间存在冲突。通过“虚拟逐头候选 + 分窗口物理驻留”，可以在不同时常驻候选并集的情况下保留 head-specific evidence，并在一个真实 128K 模型闭环中同时达到 9.9918% 常驻 KV、96.34% PPL 质量保持和 2.686x decode 加速。

当前不能表述为：

- 所有 128K 文本都能在固定 1% 预算下保持 95% 质量；
- 已经在完整 LongBench、RULER 或 Long ICL 上稳定超过 Full KV；
- 2.686x 是包含 prefill 和索引构建的端到端速度；
- 当前实现已经降低 prefill 峰值显存；
- PCA 检索本身在所有长度上都优于 block retrieval、recent cache 或其他 KV 压缩方法。

## 1. 问题定义

考虑具有 `L` 个历史 token、`H_q` 个 query heads、`H_kv` 个 KV heads 和 head dimension `D` 的自回归 Transformer。GQA 中每个 KV head 服务

```text
G = H_q / H_kv
```

个 query heads。标准 decode 在每一层都让当前 query 与全部 `L` 个历史 K/V 计算 attention：

```text
A_l,g(t) = Softmax(q_l,g(t) K_l,h^T / sqrt(D)) V_l,h
```

其中 `h` 是 query head `g` 对应的 KV head。Full KV 的存储复杂度和单步 attention 复杂度分别为：

```text
Memory_full = O(L * H_kv * D)
Compute_full = O(L * H_q * D)
```

目标是在不读取未来 token、不使用答案标签的条件下，为每个生成位置找到一个小候选集合 `I_l,g(t)`：

```text
|I_l,g(t)| << L
```

并只在该集合上执行精确 attention。同时，索引、候选 K/V 和运行时目录的 GPU 常驻字节总量必须显著低于完整 FP16 K/V。

本研究预先采用三个工程门槛：

| 指标 | 门槛 |
|---|---:|
| GPU 常驻 KV/索引比例 | 1%--10% |
| 质量保持 | 不低于 Full 的 95% |
| 真实 decode 加速 | 不低于 2.5x |

PPL 质量保持定义为：

```text
Quality retention = PPL_full / PPL_sparse
```

因此数值越接近 100% 越好；超过 100% 表示该有限样本上 sparse PPL 低于 Full，并不自动意味着方法稳定优于 Full。

## 2. 为什么常见方案在 128K 下不够

### 2.1 固定 block/page 会产生物理放大

如果 token 级 top-2% 候选分散在整个历史中，把命中 token 扩展成固定页面会显著放大精确 K/V。sports/medicine 的实测结果为：

| 主题 | token 粒度 | page=16 | page=64 |
|---|---:|---:|---:|
| sports | 2.00% | 13.78% | 30.39% |
| medicine | 2.00% | 12.69% | 26.54% |

这说明固定 page 虽然便于连续访存，却无法同时满足低于 10% 的常驻显存目标。

### 2.2 GQA 共享候选会丢失逐头证据

一个 KV head 对应四个 query heads。最直接的压缩方式是对四个 query heads 的近似分数取平均，再选择一个共享 top-k 集合。该方案在 32K 上表现良好，但在 128K 上固定 PCA32 shared top-2% 的质量保持只有 78.07%。

失败原因不是简单的“预算太小”。把 shared 候选从 2% 扩大到 3%，PPL 反而从 24.1930 恶化到 25.9052。近似索引中的假阳性被直接送进最终 softmax，增加候选既加入可能有用的 token，也加入更多干扰 token。

### 2.3 同时驻留全部逐头候选会破坏存储目标

为每个 query head 独立选择 1% token 可以恢复质量，但四个 query heads 的候选并集通常大于 3.2%。如果把四组候选同时放入 GPU，PCA64 INT4 + per-head 1% 的常驻比例为 10.999%，超过预设的 10% 门槛。

关键观察是：**逻辑上每个 query head 需要独立候选，不等于物理上这些候选必须同时驻留。** 这构成当前方法的核心设计空间。

## 3. 方法概览

当前方法由四个相互配合的组件构成：

```text
Full historical K/V on pinned CPU memory
                  |
                  v
Per-layer, per-KV-head PCA64 INT4 index on GPU
                  |
                  v
Independent top-1% virtual candidates for every query head
                  |
                  v
Two-query-head residency windows within each GQA group
                  |
                  v
Fused hash/LRU miss fill into a shared 3.2% exact-KV GPU cache
                  |
                  v
Ragged exact attention over cache-slot indices
```

其中：

- PCA 索引只做候选召回，不替代最终精确 attention；
- 每个 query head 拥有独立的逻辑 token 列表；
- 同一 GQA 组每次只处理两个 query heads；
- 两个 1% 列表直接拼接，最坏为 2%，可以放入 3.2% 热缓存；
- 计算完当前两个 heads 后复用相同物理缓存处理下一对 heads；
- 完整精确 K/V 始终保存在 pinned CPU memory，cache miss 时按 token 写入 GPU slot。

## 4. PCA64 INT4 压缩索引

### 4.1 子空间构建

对第 `l` 层、第 `h` 个 KV head，从历史 key 中采样集合 `S_l,h`，估计二阶矩：

```text
C_l,h = (1 / |S_l,h|) * sum_i K_l,h,i^T K_l,h,i
```

取最大的 `d` 个特征向量组成投影矩阵：

```text
U_l,h = TopEigenvectors(C_l,h, d)
```

主配置使用 `d=64`。历史 key 和当前 query 被投影到同一子空间：

```text
zK_l,h,i = K_l,h,i U_l,h
zq_l,g,t = q_l,g,t U_l,h
```

### 4.2 INT4 量化与候选分数

投影后的 key 按组量化为 INT4，并保存必要 scale。当前 query 也在融合 kernel 中完成投影和量化，近似检索分数为：

```text
s_l,g,t,i = QuantizedDot(zq_l,g,t, zK_l,h,i)
```

随后对每个 query head 独立执行：

```text
I_l,g(t) = TopK_i(s_l,g,t,i, ceil(r_t * L))
```

主结果使用 `r_t=1%`。索引只决定候选位置，最终输出仍读取这些位置的精确 K/V：

```text
A_l,g(t) = ExactAttention(q_l,g(t), K_l,h[I_l,g(t)], V_l,h[I_l,g(t)])
```

### 4.3 索引存储比例

完整 FP16 K/V 每个 token、每个 KV head 约占 `4D` bytes；`d` 维、`b` bit 的 key-only 索引主体约占 `d*b/8` bytes。因此忽略 scale 和 basis 时：

```text
r_index ~= d*b / (32D)
```

当 `D=128, d=64, b=4` 时，主体比例约为 6.25%；计入 FP16 scale 和 PCA basis 后实测约为 6.64%。

### 4.4 索引选择证据

| 索引 | 索引/完整 KV | 2% retained mass | candidate recall@top2 |
|---|---:|---:|---:|
| QAbs16 FP16 | 50.00% | 0.8437 | 0.8775 |
| PCA32 INT8 | 6.64% | 0.8522 | 0.8908 |
| PCA48 INT8 | 9.77% | 0.8652 | 0.9587 |
| PCA64 INT4 | 6.64% | 0.8661 | 0.9588 |
| PCA96 INT4 | 9.77% | 0.8708 | 0.9807 |

PCA64 INT4 在接近 PCA32 INT8 存储成本的情况下把 candidate recall 提高到 0.9588，是当前质量、存储和扫描开销之间的折中点。

## 5. Per-head 虚拟候选

### 5.1 从 KV-head 共享到 query-head 独立

共享方案为每个 KV head 选择一个集合：

```text
I_l,h = TopK_i(mean_g s_l,g,i)
```

当前方案改为：

```text
I_l,g = TopK_i(s_l,g,i),  g in GQAGroup(h)
```

这些集合是“虚拟候选”：它们是 query head 的逻辑 attention 索引，不要求对应 K/V 在生成这一时刻全部同时常驻 GPU。

### 5.2 128K 逐头候选消融

在同一个 128K religion 样本前 32 个目标 token 上，Full PPL 为 15.1605：

| 配置 | PPL | PPL/full | 质量保持 | 常驻 KV |
|---|---:|---:|---:|---:|
| PCA32 INT8，per-head 直接 1% | 21.5557 | 1.4218 | 70.33% | 10.985% |
| PCA32 INT8，召回 2% 后精确重排到 1% | 15.9499 | 1.0521 | 95.05% | 15.317% |
| PCA64 INT8，per-head 直接 1% | 15.8283 | 1.0440 | 95.78% | 17.249% |
| PCA64 INT4，per-head 直接 1% | 15.7358 | 1.0379 | **96.34%** | 10.999% |
| exact-QK oracle，per-head 1% | 15.8072 | 1.0427 | 95.91% | 不可部署 |

PCA64 INT4 的结果接近 exact-QK 1% oracle，说明当前主要突破来自恢复 query-head 独立性，而不是不断扩大 shared 候选。

PCA48/PCA56 INT4 的质量保持分别只有 92.20% 和 93.95%，未达到 95% 门槛。因此当前版本固定 PCA64，不再继续通过减少投影维度换取表面上的存储下降。

## 6. Residency-windowed GQA

### 6.1 核心算法

设一个 KV head 对应四个 query heads，每个 query head 的候选预算为 1%。当前主配置每次处理两个 heads：

```text
for each layer l:
    candidate_ids = PCA64_INT4_scan(all_query_heads)

    for (g0, g1) in GQA_pairs:
        ids0 = candidate_ids[g0]          # top-1%
        ids1 = candidate_ids[g1]          # top-1%
        requested = concatenate(ids0, ids1)

        slots = fused_hash_lookup(requested)
        misses = requested[slots == NOT_RESIDENT]
        victims = fused_recency_select(misses.count)
        copy_exact_kv_from_pinned_host(misses, victims)
        rebuild_or_update_hash_directory()

        out[g0] = ragged_attention(q[g0], slots_for(ids0))
        out[g1] = ragged_attention(q[g1], slots_for(ids1))
```

“直接拼接”与“先显式构造并集”在数学上使用相同候选，但省去一次去重和并集 materialization。重复 token 在 hash lookup 后可以映射到相同 resident slot，不需要在 Python/PyTorch 层先整理集合。

### 6.2 为什么两个 heads 是当前最优窗口

- 单-head 窗口：候选工作集小，但要执行四轮目录更新和 attention，调度开销较高；
- 双-head 窗口：最坏工作集为 2%，可以被 3.2% 热缓存覆盖，只需两轮执行；
- 四-head 同驻留：吞吐更高，但候选并集和热缓存使常驻比例达到 10.999%；
- 1.5% 动作：双-head 最坏为 3%，仍可使用同一个 3.2% 热缓存；
- 2% 动作：双-head 最坏为 4%，需要退回单-head 流式或扩大物理缓存。

### 6.3 128K 路径比较

| 路径 | PPL | 质量保持 | 常驻 KV | 热缓存命中 | 287 步 decode | 相对 Full 加速 |
|---|---:|---:|---:|---:|---:|---:|
| Full SDPA | 15.1605 | 100% | 100% | - | 94.054 s | 1.000x |
| 四组同时驻留，4.1% 热缓存 | 15.7358 | 96.34% | 10.999% | 82.30% | 31.279 s | 3.007x |
| 单-head 流式，3.2% 热缓存 | 15.7358 | 96.34% | 9.9918% | 83.14% | 40.768 s | 2.307x |
| 双-head 显式并集，3.2% 热缓存 | 15.7358 | 96.34% | 9.9918% | 80.89% | 37.314 s | 2.520x |
| **双-head 直接拼接，3.2% 热缓存** | **15.7358** | **96.34%** | **9.9918%** | **81.52%** | **35.018 s** | **2.686x** |
| 单-head 流式，1.1% 热缓存 | 15.7358 | 96.34% | 7.8265% | 47.65% | 44.519 s | 2.113x |

双-head 直接拼接是当前 Pareto 主结果。它比四组同时驻留少约 1.01 个百分点常驻 KV，跨过 10% 门槛；相比单-head 流式，又减少了一半的窗口调度次数。

## 7. 分层物理 KV 生命周期

### 7.1 存储布局

```text
GPU resident state
  - PCA64 INT4 key index
  - PCA basis and quantization scales
  - 3.2% exact K/V hot cache
  - hash keys, hash slots, inverse token ids
  - uint8 recency state
  - current token and bounded decode reserve

Pinned CPU memory
  - full exact K/V history
```

prefill 完成后，完整 K/V 被转入 pinned CPU memory，原始 GPU full K/V tensor 被释放。之后每个 decode step 只扫描 GPU 上的压缩索引，并按 miss 把少量精确 K/V 填入 GPU 热缓存。

### 7.2 Hash/LRU 目录

目录使用开放寻址 hash table，记录逻辑 token id 到物理 cache slot 的映射。融合 CUDA 路径负责：

- resident lookup；
- 命中项 recency 更新；
- 基于 uint8 recency histogram 选择 victim slots；
- miss 写入后的目录更新或 rebuild；
- 变长 hit/miss 数量处理。

先前 PyTorch `searchsorted/topk/sort` 目录约需 0.299 ms/layer；融合实现为 0.0502 ms/layer。uint8 recency 每步衰减，命中项置为 255，在当前 256-token 轨迹内提供足够的 LRU 时间范围。

### 7.3 Miss 搬运

候选集合具有较强时间局部性。32K 五主题、256-token 顺序轨迹中，3.2% 热缓存命中率约为 78.68%--83.34%。以最差 sports 为例，每步只需从主存读取：

```text
2% * (1 - 78.68%) = 0.4264% of full K/V
```

在 128K、8 KV heads、FP16 K/V 下约为 2.184 MiB/layer。当前实现让 mapped-host miss 直接写入 GPU cache slots，再由 ragged attention 读取这些 slots，避免每个 query head 重复访问 host K/V。

### 7.4 与逻辑 mask 稀疏的区别

当前路径不是“保留完整 GPU K/V，只把 attention mask 变稀疏”。它实际执行了：

1. 清除原始 GPU full K/V；
2. 完整精确 K/V 转入 CPU；
3. GPU 常驻压缩索引和有界热缓存；
4. decode 时按候选执行 hash lookup 和 miss fill；
5. attention 只读取热缓存 slot。

因此，9.9918% 是物理 GPU 常驻状态的字节比例，不是 token mask 比例或理论 attention-link 下界。

## 8. 复杂度与成本模型

设 PCA 维度为 `d`，per-head 候选比例为 `r`，GQA 流式窗口大小为 `w`。

### 8.1 单步计算

近似扫描：

```text
O(L * H_q * d)
```

最终精确 attention：

```text
O(r * L * H_q * D)
```

目录和 miss fill 与候选数及 miss 数近似线性：

```text
O(r * L * H_q) + O(miss_count * D)
```

相比 Full 的 `O(L * H_q * D)`，收益主要来自 `d << D` 的低维扫描和 `r << 1` 的最终 attention。实际速度还受到 kernel launch、top-k、PCIe miss、模型非 attention 层以及多 GPU 同步影响。

### 8.2 常驻存储

```text
M_resident = M_PCA_index + M_exact_hot + M_directory + M_decode_reserve
```

当前主结果的总比例为 9.9918%。PCA64 INT4 约占 6.64%，精确热缓存目标为 3.2%，其余部分由 basis、scale、目录和运行时 reserve 共同构成。因为统计按真实 tensor bytes 进行，不能简单把名义比例相加后当作最终数字。

### 8.3 Setup 摊销

128K 主结果中：

| 阶段 | Full | 当前方法 |
|---|---:|---:|
| prefill | 219.954 s | 214.659 s |
| cache 转换 | 0 | 24.927 s |
| 287 步 decode | 94.054 s | 35.018 s |
| 总计 | 314.008 s | 274.604 s |

因此：

```text
Decode speedup = 94.054 / 35.018 = 2.686x
Protocol total speedup = 314.008 / 274.604 = 1.143x
```

cache 转换是当前端到端速度的主要额外成本之一。未来需要在 chunked prefill 中直接建立 PCA 索引并把精确 K/V 落到 host，避免“先建立 Full GPU KV，再整体转换”。

## 9. 实验结果

### 9.1 主结果

**模型：** Llama-3.1-8B-Instruct  
**上下文：** 128K religion 文本流  
**目标区间：** 前 32 个 target tokens；加 256 query warm-up 后共计 287 个逐 token decode step  
**硬件：** 4 x RTX 3090  
**计时：** Full 与 sparse 都在每个 token forward 前后执行全设备同步  
**方法：** PCA64 INT4、per-head top-1%、双-head 直接拼接、3.2% 热缓存

| 指标 | 结果 |
|---|---:|
| Full PPL | 15.1605 |
| Sparse PPL | 15.7358 |
| PPL/full | 1.0379 |
| 质量保持 | **96.34%** |
| GPU 常驻 KV/索引 | **9.9918%** |
| 热缓存命中率 | 81.52% |
| Full decode | 94.054 s |
| Sparse decode | 35.018 s |
| Decode 加速 | **2.686x** |
| 含 prefill/转换总加速 | **1.143x** |

这是当前唯一可以同时写入标题级结果的 Pareto 点。

### 9.2 长度缩放证据

早期 PCA32 shared top-2% 物理路径在相同协议下得到：

| 长度 | PPL/full | 常驻 KV | Decode 加速 | 总加速 |
|---:|---:|---:|---:|---:|
| 32K | 1.0054 | 9.9750% | 1.531x | 1.163x |
| 64K | 0.9885 | 9.9735% | 2.366x | 1.361x |
| 128K | 1.2808 | 9.9727% | 2.340x | 1.231x |

该结果说明速度收益随长度增长而兑现，同时也暴露 shared 候选在 128K 的质量拐点。后续 per-head residency-windowed GQA 正是对该失败的修复。

### 9.3 独立 computer 主题

在主配置冻结后，使用此前 cache-trace 统计未覆盖的 computer/comp.graphics 主题复验：

| 动作 | PPL | Full PPL | 质量保持 | 常驻 KV | Decode 加速 |
|---|---:|---:|---:|---:|---:|
| per-head 1%，双-head 流式 | 65.7822 | 60.4325 | 91.87% | 9.9918% | **2.761x** |
| per-head 1.5%，双-head 流式 | 60.5298 | 60.4325 | **99.84%** | 9.9918% | 2.278x |
| per-head 2%，单-head 流式 | 57.1567 | 60.4325 | 105.73% | 9.9918% | 1.528x |

这组结果非常重要：物理存储结构可以容纳更高逻辑预算，而不增加常驻 KV；但质量和速度之间需要位置级控制。固定 1% 是快速动作，不是全局安全动作。

### 9.4 LongBench 初步质量探针

在 7.5K 上下文、四类 LongBench 任务、每任务 5 条的配对探针中，使用 PCA64 INT4、per-head 2.5%、3.2% 热缓存：

| 任务 | Full KV | 当前方法 | KV ratio |
|---|---:|---:|---:|
| NarrativeQA | 27.68 | 31.42 | 10.46% |
| HotpotQA | 34.67 | 34.67 | 10.44% |
| PassageRetrieval | 20.00 | 20.00 | 10.36% |
| LCC | 68.90 | 81.20 | 11.25% |
| 四任务平均 | 37.81 | 41.82 | 10.63% |

20 个配对样本中，当前方法 2 个更好、17 个相同、1 个更差。该小样本只说明方法能够进入真实任务生成且暂未出现系统性质量塌陷，不能证明稳定超过 Full。

短上下文速度当前不合格：Full online latency 为 0.857 s，当前逐 token suffix 实现为 10.081 s。原因是 7.5K 下 attention 占比小、cache 转换不可摊销，并且 question suffix 被逐 token 回放。该结果不影响 128K 核心机制，但要求最终系统增加短上下文 Full fallback 或 chunked sparse suffix。

## 10. 关键消融与负结果

### 10.1 不能只扩大 shared 候选

| 配置 | 128K PPL | 结论 |
|---|---:|---|
| PCA32 shared 2% | 24.1930 | 原始失败点 |
| PCA32 shared 3% | 25.9052 | 候选更多但干扰更强 |
| PCA64 shared 2% | 27.1153 | 增加投影维度没有修复 shared 结构 |

### 10.2 PCA32 oversampling 可以恢复质量，但存储过高

PCA32 先召回 per-head 2%，再用精确 QK 重排到 1%，质量保持可达 95.05%，但四组候选同时驻留时常驻 KV 为 15.317%。这验证了“近似召回假阳性”问题，却不满足存储目标。

### 10.3 共享均值在中等长度有效，但不是 128K 最终方案

PCA32 + shared-mean + 固定 2% 在三个 32K 窗口的合并质量保持为 96.89%，在独立 religion-window0 上达到 99.47%，128K attention 微基准为 3.41x。它是有价值的中等长度动作，但 128K 下 head-specific evidence 缺失使其退化到 78.07%。

### 10.4 更小 PCA 维度没有通过质量门槛

PCA48/PCA56 INT4 虽然降低索引成本，但 128K 质量保持分别只有 92.20%/93.95%。继续压缩索引维度不是当前最有效方向。

### 10.5 更小热缓存不一定更快

1.1% 热缓存把常驻比例降到 7.8265%，但命中率降至 47.65%，decode 加速只有 2.113x。低存储和低延迟并不总是一致；PCIe miss 与目录更新会抵消显存节省。

### 10.6 Token 类型规则不足以决定预算

数字、专名和低频 token 平均更可能需要远程证据，但并非所有数字都危险，也并非所有功能词都安全。可靠预算控制需要结合 provisional confidence、entropy、margin、检索 gap、候选稳定性、上下文长度和历史状态，而不能只看下一个 token 的词法类别。

## 11. 与旧 LongBench block 方法的关系

旧方法和当前方法解决的是相邻但不相同的问题：

| 维度 | 旧 question-aware block 方法 | 当前 per-head hierarchical PCA 方法 |
|---|---|---|
| 主要信号 | query 与 block 的文本/结构匹配 | 模型内部 Q/K 子空间相似度 |
| 粒度 | block/page 或任务 action | layer、KV head、query head、token |
| 主要长度 | LongBench 约 7.5K | 64K--128K |
| 执行方式 | 一次性 block 选择，部分版本含 direct operator | 真实物理 KV 分层和 sparse attention |
| 优势 | 短任务选择成本低，query-aware 语义强 | 与模型状态耦合，逐头证据，更适合超长 decode |
| 主要风险 | RAG 边界、任务规则和 direct operator 混杂 | 索引构建、PCIe miss、短上下文固定开销 |

当前技术报告只把 per-head hierarchical PCA 作为主方法。未来可以把 block selector 用作粗粒度 scope prior，但必须作为独立组件做消融，不能把文本 RAG 的收益隐含写成纯 KV compression 的收益。

## 12. 当前方法的创新点

### 12.1 Query-head 虚拟候选与 KV-head 物理存储解耦

现有 GQA cache 通常让同一 KV head 的 query heads 共享保留 token，或者为逐头候选支付完整并集存储。当前方法把逻辑 attention 集合与物理驻留集合分离：每个 query head 保留独立证据，但物理 K/V 按时间窗口复用。

### 12.2 Residency-windowed GQA

GQA 不再只是模型结构约束，而成为运行时调度维度。通过控制同时活跃的 query-head 数量，可以在候选并集、目录开销、热缓存大小和 kernel 调度之间取得可测量的 Pareto 折中。

### 12.3 压缩索引、精确 K/V 和动态目录的完整闭环

方法不是只做低维近似 attention。PCA INT4 只负责全局搜索，最终 attention 使用精确 K/V；完整 K/V 在 CPU，GPU 热缓存由 fused hash/LRU 管理。该结构同时处理召回质量、物理显存和在线搬运成本。

### 12.4 从失败诊断推导方法，而非参数枚举

128K shared 候选失败后，扩大预算和扩大 PCA 维数都没有恢复质量。逐头候选恢复到接近 exact-QK oracle，直接证明缺失的是 head-specific evidence；residency-windowed GQA 随后解决逐头候选的物理并集问题。方法链条具有清晰的因果诊断基础。

## 13. 当前局限

1. **主结果样本量不足。** 9.9918%/96.34%/2.686x 只来自一个 128K 主题窗口的前 32 个 target tokens。
2. **固定预算不通用。** 独立 computer 主题表明 1% 会低于 95% 质量门槛，至少需要 1.5% 或位置级升级。
3. **端到端加速有限。** 主结果 decode 为 2.686x，但包含 prefill 和转换后只有 1.143x。
4. **峰值显存尚未降低。** 当前转换先建立 full GPU KV，转换瞬间仍同时持有 full KV 和部分层次化状态。
5. **短上下文路径低效。** 4K--8K 下 attention 计算占比不足，逐 token suffix 和 kernel 调度使 sparse 路径慢于 Full。
6. **缺少多模型验证。** 当前物理主结果集中在 Llama-3.1-8B-Instruct 和 RTX 3090。
7. **缺少 batch throughput。** 现有计时主要是 batch=1 latency，尚未证明服务场景吞吐收益。
8. **外部任务证据不足。** LongBench 只有 20 条初步配对样本，RULER、Long ICL 和长生成尚未完成同一物理闭环。
9. **CPU/PCIe 依赖需要单独报告。** 不同主机内存带宽、NUMA 和 PCIe 拓扑可能显著影响 miss fill。

## 14. 下一步实验优先级

### P0：冻结主方法，扩大独立验证

- 固定 PCA64 INT4、3.2% 热缓存和 residency-windowed GQA；
- 在未参与方法选择的多个 128K 主题和不重叠窗口上复现；
- 每个窗口至少评估 256 个 target tokens，而不是 32 个；
- 同时报告均值、最差窗口、bootstrap 置信区间和逐位置 NLL gap。

### P1：训练因果安全预算控制器

动作集合建议为：

```text
fast:  per-head 1.0%, two-head window
safe:  per-head 1.5%, two-head window
high:  per-head 2.0%, one-head window
```

router 只能使用当前可见信号：上下文长度、provisional logits、entropy、margin、PCA gap、top-k 稳定性、cache miss rate 和历史动作。训练使用 train/calibration 主题产生的 counterfactual labels，最终阈值冻结后在新主题真实混合轨迹上测试。

### P2：消除转换与峰值显存

- chunked prefill 时在线更新 PCA covariance/basis；
- 每个 chunk 的精确 K/V 直接写入 pinned host；
- GPU 只保留当前 chunk、索引构建状态和热缓存；
- 报告 peak allocated、peak reserved、host bytes 和 setup latency。

### P3：完善 benchmark

- LongBench 16 tasks，matched Full，至少 M100；
- RULER 32K/64K/128K；
- Long ICL 和长代码生成；
- 第二模型和不同 GQA ratio；
- batch=1/2/4 的 latency 与 throughput；
- 对比 Full KV、recent、H2O、SnapKV、PyramidKV、AdaKV 及强 question-aware block baseline。

### P4：短长路径统一

建议的最终 runtime policy：

```text
if context_length < L0:
    use Full SDPA or one-shot block selection
else:
    use per-head hierarchical PCA cache

within long-context decode:
    route each position among 1.0%, 1.5%, and 2.0%
```

`L0` 必须通过真实端到端测量确定，而不是根据 attention 理论比例手工设定。

## 15. 面向论文的实验表结构

### 15.1 主表

| Method | Quality | Resident KV | Peak GPU | Decode speed | Total speed | Host traffic |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 待填 | 100% | 待填 | 1.00x | 1.00x | 0 |
| Recent | 待填 | 待填 | 待填 | 待填 | 待填 | 0 |
| AdaKV/SnapKV 等 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ours shared | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ours per-head simultaneous | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ours residency-windowed | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| Ours + risk router | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

### 15.2 必要消融

- PCA16/32/48/56/64/96；
- INT4/INT8；
- shared-mean/shared-max/per-head；
- stream group size 1/2/4；
- exact hot cache 1.1%/2.1%/3.2%/4.1%；
- fixed 1%/1.5%/2% 与动态 router；
- sorted directory 与 fused hash/LRU；
- host-direct attention 与 miss-fill-then-attention；
- 只报告逻辑 token ratio与报告真实 tensor bytes 的差异。

## 16. 复现入口

主要代码：

- `ymluo/projects/qwen3_top2_head_limit3_ppl/src/hierarchical_pca_cache_20260715.py`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/src/qabs_cuda_kernels.py`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_hierarchical_longbench_probe_20260715.py`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_shifted_dynamic_physical_cache_ppl_20260715.py`

主要研究记录：

- `ymluo/doc/section206_位置级证据预算与PCA索引_20260715.md`

实现核查至少应包含：

1. INT4 扫描 kernel 与整数参考逐项一致；
2. hash lookup、victim selection 和 rebuild 与 CPU 参考一致；
3. mapped-host miss fill 后 K/V 与源 tensor 一致；
4. ragged attention 与相同候选上的 dense reference 一致；
5. Full 与 sparse 使用相同 prompt、target、模型映射和同步计时协议；
6. resident ratio 按真实 tensor bytes 统计；
7. 不把首次 JIT 编译时间混入稳定态速度；
8. 不使用目标 token、答案或测试主题调预算阈值。

## 17. 总结

当前工作的核心进展不是又找到一个更复杂的 block scorer，而是识别并解决了超长上下文 GQA 稀疏化中的结构冲突：

```text
query heads 需要不同证据
            +
KV heads 在物理上共享 K/V
            +
全部逐头候选的并集超过显存预算
```

PCA64 INT4 提供低成本全局候选召回，per-head 虚拟候选保留 head-specific evidence，residency-windowed GQA 则把逻辑逐头独立性映射到有界物理工作集。三者共同形成了当前主结果：128K、9.9918% 常驻 KV、96.34% 质量保持、2.686x decode。

这已经足以支持一条有明确机制、失败诊断和真实系统闭环的 ICLR 主线，但还不足以支持最终论文结论。后续工作的重点应从继续枚举静态参数转向三件事：扩大冻结方法的独立验证、训练因果安全预算控制器，以及在 chunked prefill 中消除转换和峰值显存。只有这三项完成后，当前单点 Pareto 结果才能升级为稳定、通用且可复现的长上下文 KV 系统。
