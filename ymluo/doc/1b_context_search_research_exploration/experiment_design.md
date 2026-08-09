# 1B Token Context 搜索：可证伪实验设计

**文档类型：** falsification and profiling plan  
**注意：** E1-E6 的实验已经先于本文运行；下面的通过/失败条件是为了审计现有证据而追补的，不是预注册标准。E7-E11 是后续实验，应在运行前冻结协议。

## 1. 实验目标

本轮不是直接证明“1B token context 已经可以高效问答”，而是依次回答五个更小的问题：

1. 一个 block 是否可以用一个 K-mean 做全局语义检索？
2. block 内 token K 是否存在可压缩的低秩结构？
3. 少量无监督 K 原型是否能保留 exact max-QK？
4. 真实长文档的 residual K 是否具有位置局部性？
5. 这种局部性是否能减少 seed 后 KV centroid 近邻扩展的点积？

最终因果链必须是：

```text
结构属性存在
-> 对应算法确实利用该属性
-> attention 候选没有严重损失
-> 最终生成质量保持或提高
-> wall-clock / memory / communication 成本下降
```

当前实验只完成了前两段的一部分。

## 2. 公共实验设置

### 2.1 模型与向量

| 项 | 值 |
|---|---|
| 模型 | Qwen/Qwen3-0.6B |
| block size | 256 tokens |
| head dimension | 128 |
| 通道 | L3/H10、L21/H8、L6/H7、L16/H14 |
| 向量来源 | 真实模型 causal forward 的 Q/K |
| 合成向量 | 不使用 |
| 索引 dtype | FP16 |

四个通道是此前选定的代表性通道，不代表全层全 head。任何跨 head 的普遍性结论都属于证据不足。

### 2.2 两份10M语料

**MuSiQue 10M：**

- 9,999,872 tokens；
- 39,062 blocks；
- 2,000 个二跳 step queries；
- 文本段落真实，但全局 block 顺序被打乱；
- 用于有 gold evidence label 的直接检索，以及局部性的负对照。

**Real LongBench 10M：**

- 9,999,872 tokens；
- 39,062 blocks；
- 982 个真实 records；
- 来自 2WikiMQA、HotpotQA、MuSiQue、Qasper、NarrativeQA、MultiFieldQA、QMSum、GovReport；
- 保留 record 内原始顺序；
- 同时测试 block-local forward 和完整 record causal prefill。

### 2.3 数据泄漏检查

每个输出必须记录：

```text
contains_synthetic_vectors = false
selection_uses_gold = false  # 适用时
model / block count / token count / pair specs / vector space
```

gold label 只允许用于最终计算 rank/recall，不能用于索引构建、通道内候选选择或树结构构建。

## 3. 指标定义

### 3.1 Gold evidence Recall@K

```text
Recall@K = mean[gold evidence block rank <= K]
```

该指标用于自然语言/step query 到证据 block 的检索。它不能与 centroid neighbor recall 混用。

### 3.2 Exact max-QK Top1 agreement

对同一组 candidate blocks：

```text
exact winner  = argmax_block max_token,query,profile q^T k
approx winner = argmax_block max_prototype,query,profile q^T p
agreement     = mean[exact winner == approx winner]
```

该指标只判断压缩 K 是否保持 exact max-QK 排名，不直接判断 gold evidence。

### 3.3 Exact K-centroid Top10 neighbor recall

```text
oracle10 = full_scan_top10(v_query @ all_block_centroids)
found10  = hierarchical_top10(v_query)
recall   = |oracle10 intersect found10| / 10
```

这是 seed 后 KV 几何扩展的代理指标，不是 QA accuracy，也不是 attention-token recall。

### 3.4 估算点积加速

```text
dot_fraction = 1/parent_size + actual_leaf_fraction/leaf_size
               + actual_block_scan_fraction
speedup = 1/dot_fraction
```

这里只统计与512维 block/leaf/parent向量的等价点积数量。不得写成 wall-clock 加速。

### 3.5 证据状态

每个实验只能标记为：

- **通过：** 达到预先写明的可用条件，且对照支持对应因果解释；
- **失败：** 指标直接违反可用条件；
- **部分支持：** 先验存在，但对应算法或目标阈值未通过；
- **证据不足：** benchmark、样本、对照或下游指标无法回答该问题。

## 4. E1：K-mean 是否是可用的 block 语义地址

### 猜想

同一 block 内 K 方向足够集中，而且不同 block 的 K-mean 具有足够区分度，因此当前 Q 可以直接在39,062个 K-mean 中找出 gold evidence。

### 操作化

1. 计算每个 block、每个通道的 concentration。
2. 比较随机 block K-mean cosine 与相邻 block cosine。
3. 捕获每个 step query 末尾16个真实 Q。
4. 比较 last/mean/max Q、raw/cosine、global-centered、1/4 centroids、pre/post-RoPE 共13种方法。
5. 每种方法全库打分，保存前512个 block 和 gold rank。

### 参数

| 参数 | 值 | 原因 |
|---|---:|---|
| queries | 2,000 | dev/test、bridge/answer 各500 |
| query tokens | 16 | 覆盖当前 step 状态末尾短窗口 |
| candidate cap | 512 | 同时观察极小预算和宽松预算 |
| segment means | 4 | 检查单均值是否因内部多模态失败 |
| random block pairs | 200,000 | 稳定估计全局各向异性 |

### 通过/失败条件

- “K 存在公共方向”通过：concentration 明显高于随机单位向量基准，且跨 block cosine 高。
- “K-mean 可替代全局检索”严格通过：test Recall@16 至少80%，并在 bridge 与 answer 两类都成立。
- 失败：任一类 test Recall@16 低于50%，说明16-block读取预算不可用。
- 证据不足：只测 concentration，不测 gold rank；或者只在 train/dev 选择最优方法。

### 逐阶段 profiling

| 阶段 | 检查 | artifact | 失败解释 |
|---|---|---|---|
| K capture | block 数、通道、concentration | index `summary.json` | hook/shape/space 错误 |
| geometry | random/adjacent cosine | `geometry_v2.json` | 公共方向或局部性假设错误 |
| query capture | 2,000 x 16 Q | retrieval `summary.json` | query 太短或状态文本错误 |
| ranking | 每方法 target rank | `rows.jsonl` | mean 表征缺少证据方向 |
| final | Recall@K | 本地 evidence snapshot | 判断能否进入下游预算 |

### 当前判定

公共方向通过；全局语义地址失败。结果见 [visualization_results.md](visualization_results.md#3-e1结果k-mean-集中但不能全局检索)。

## 5. E2：block residual K 是否中低秩

### 猜想

减去 block token mean 后，256个 token K 的主要能量可由远小于128的子空间表示，并且该现象在不同语料和完整 record causal prefill 中稳定。

### 操作化

1. 固定随机 seed 抽样 blocks。
2. 对每个 block、每个通道构造 `K_res = K - mean_token(K)`。
3. 执行 SVD。
4. 保存 rank-8/16/32 energy、rank90、rank95 和 effective rank。
5. 比较 MuSiQue、LongBench block-local、LongBench full-record。

### 参数

| 参数 | 值 | 原因 |
|---|---:|---|
| full-record sample blocks | 512 | 原始 K 分片读取和 SVD 成本可控 |
| seed | 23 | 固定可复现样本 |
| target ranks | 8/16/32 | 对应32x/16x/8x basis压缩粒度 |
| energy threshold | 90%/95% | 描述谱，不作为 max-QK 安全阈值 |

### 通过/失败条件

- 强 rank-16 假设通过：四个通道的 mean rank-16 residual energy 都达到90%。
- 中低秩先验通过：四个通道 rank90 都不超过32，且 block-local/full-record 的范围一致。
- 失败：任一代表通道 rank90 接近128，或完整 record 前向显著破坏低秩。
- 证据不足：只测全局 centroid spectrum，不测 block 内 token residual。

### 当前判定

强 rank-16 假设未完全通过；中低秩先验通过。rank-16 只能作为有损压缩候选，不能直接宣称保留90%或安全保持 attention。

## 6. E3：无监督 FPS 原型能否近似 exact max-QK

### 猜想

每个 block 保留16个在欧氏距离上覆盖原始 K 的真实 token 原型，可以近似256个 token 的支持函数 `max q^T k`。

### 操作化

1. 对每个 block、每个通道执行 nested farthest-point sampling。
2. 保存前1/2/4/8/16个原型和覆盖半径。
3. 对每题已有的16个 lexical candidate blocks，分别计算 exact max-QK 和 FPS approximate max-QK。
4. 计算 gold recall、exact Top1 agreement、score gap。
5. 用 `||q||rho` 上界计算要安全保留的候选比例。

### 通过/失败条件

- 可用近似通过：FPS-16 test Top1 agreement 至少90%。
- 安全剪枝通过：保持 exact Top3 recall=100% 时，平均保留候选不超过25%。
- 失败：Top1 agreement低于50%，或安全上界必须保留全部候选。
- 证据不足：只报告压缩率，不与 exact max-QK 比较。

### 当前判定

失败。bridge/answer test Top1 agreement 为14.2%/22.0%，安全上界保留100%候选。失败的是无监督欧氏覆盖这一操作化，不是所有 query-aware K 压缩方法。

## 7. E4：真实 record 是否具有 residual-K 局部连续性

### 猜想

真实文档内相邻 blocks 的 global-centered K centroid 比跨 record 随机 blocks 更相似，并且相关性随距离衰减；打乱 block 顺序后该结构消失。

### 操作化

1. 对每个通道减去全语料 block mean并归一化。
2. 在每个 LongBench dataset 内计算同 record offset `1/2/4/8/16/32` 的 cosine。
3. 抽取跨 record random pairs 作为背景。
4. 拟合背景以上相关性的指数长度。
5. 在全局打乱 MuSiQue 上重复 offset 分析。

### 参数

| 参数 | 值 | 原因 |
|---|---:|---|
| random pairs | 100,000 | 每 dataset 估计 cross-record 背景 |
| offsets | 1/2/4/8/16/32 | 覆盖256至8192 token距离 |
| datasets | 8 | 检查局部性是否只存在于一种文体 |

### 通过/失败条件

- 局部性通过：八个 dataset 的 mean adjacent cosine 都至少比 cross-record random 高0.10。
- 距离结构通过：多数通道在 offset 增加时总体下降。
- 对照通过：打乱 MuSiQue global-centered adjacent cosine 接近其 random baseline，绝对差小于0.02。
- 失败：局部性只由跨文档公共方向造成，去中心后消失。

### 当前判定

通过。八个数据集的相邻优势约为0.238到0.515；打乱对照约为0。NarrativeQA/QMSum跨 record 背景也较高，因此必须使用“相邻减去跨 record 背景”，不能只看 raw adjacent cosine。

## 8. E5：固定两级树能否利用局部连续性

### 猜想

连续位置 group 的 centroid 能代表其中 blocks，因此先搜索 parent/leaf 再精扫少量 blocks，可以找回全库 exact residual-K centroid 近邻。

### 操作化

算法以 [design.md](design.md#53-stage-c两级-residual-k-centroid-搜索) 的 Stage C 为准。对 block-local LongBench、full-record LongBench 和打乱 MuSiQue 使用完全相同参数。

### 通过/失败条件

- 可用 tradeoff 通过：最终精扫5% blocks时，full-record Top10邻居召回至少70%，估算点积减少至少5倍。
- 高压缩目标通过：最终精扫2% blocks时，召回至少70%，估算点积减少至少10倍。
- 结构因果对照通过：相同参数下，真实 LongBench recall 比打乱 MuSiQue 高至少30个百分点。
- 失败：真实数据与打乱对照相近，或完整 record K 明显低于可用阈值。

### 当前判定

- 可用 tradeoff：通过，full-record 为72.78% recall、7.79x估算点积减少。
- 高压缩目标：失败，full-record 为52.86% recall、16.44x估算点积减少。
- 结构对照：通过，对应打乱对照仅27.16%和12.80%。

因此当前结论是“存在可调速度-召回折中”，不是“16倍加速且高召回已经同时成立”。

## 9. E6：block-local 结果是否是前向方式伪影

### 猜想

如果局部性和低秩来自模型本身与文档结构，那么在完整 record causal prefill 中仍应存在；如果只来自每256 tokens重新开始位置和上下文，它们会明显消失。

### 操作化

- 对同一39,062个 LongBench blocks 分别构建 block-local 与 full-record K mean；
- 比较配对 raw/global-centered centroid cosine；
- 独立比较 residual rank8/16/rank90；
- 用同一层次搜索协议比较 exact-neighbor recall。

### 通过/失败条件

- 低秩稳定：full-record rank90 仍不超过32且范围与 block-local 接近。
- 局部性稳定：full-record adjacent residual cosine仍显著为正。
- 层次可用：full-record在5%精扫下达到70% recall。
- 失败：只有block-local满足上述条件。

### 当前判定

低秩、局部性和5%层次搜索均通过；逐 block residual方向的一致性在四通道间差异较大，因此不能声明两种前向得到逐 block 等价 K。

## 10. E7：RAG seed 加 KV 扩展是否改善真实多步生成

**状态：运行前协议，当前最高优先级。**

### 对照

```text
A. RAG only
B. RAG seed + residual-K parent/leaf expansion + post-RoPE exact rerank
C. RAG seed + flat exact K rerank  # 质量 oracle，规模允许时
```

### 固定条件

- 同一500条未参与调参的二跳问题；
- 同一第一次 RAG seed 候选和顺序；
- 同一 bridge 生成模型、提示词、随机种子和解码参数；
- A/B 使用相同累计读取 token budget；
- 所有阈值只在 train/dev 冻结。

### 分阶段输出

| 阶段 | 指标 | 必须保存的 debug artifact |
|---|---|---|
| first seed | gold first-block rank/recall | 每题 seed blocks 与分数 |
| state update | bridge/entity exact match与文本 | 生成token、解析结果、失败原因 |
| KV expansion | second evidence rank/recall | parent/leaf/candidate ids与分数 |
| exact rerank | attention top-token recall/mass | exact post-RoPE QK top tokens |
| reader | Answer@128、Final F1/EM | 完整生成文本 |
| runtime | capture/index/search/load/decode ms | 每阶段CUDA timing与通信字节 |

### 通过/失败/证据不足

- 方法质量通过：B相对A在配对最终答案上有显著净胜，或在统计等价质量下显著减少读取tokens/延迟。
- KV阶段通过：给定相同正确seed与bridge时，B显著提高第二证据或attention token recall。
- 失败：centroid邻居召回提高但第二证据、attention或答案均不提高。
- 证据不足：B使用更多reader tokens、不同RAG seed或不同生成模型，导致无法归因。

## 11. 后续实验 E8-E11

### E8：自适应 change-point segment tree

用 residual centroid 的相邻变化量自动决定 segment 边界，与固定8/64树比较。在相同估算点积预算下，若 exact-neighbor recall没有提高，则否定“固定边界是主要瓶颈”。

### E9：query-aware support coreset

从train生成Q学习每个block的K原型或方向上界；严格在held-out query上比较FPS-16。通过条件是Top1 agreement显著超过FPS且安全候选保留率低于50%。

**当前结果：严格安全版本失败，概率代理在完整raw block轴通过。** 128个train-only查询方向原型在512个均匀抽样真实blocks上得到0.819平均分数Spearman；完整39,062-block轴上，2,048/4,096/9,766候选对raw exact Top16的覆盖为89.51%/95.12%/98.80%。但是对z-score exact Top16，相同预算只有63.94%/76.89%/90.50%。Lipschitz安全界平均保留99.9997%候选。下一步只继续概率路由加exact精排，不再把当前原型界称为安全剪枝。

### E10：全层全head属性与路由

**E10a 64题探索和480题扩展复验均已完成。** 当前冻结结果包含28层、每层16个query heads、480个真实queries和每head Top16 blocks；扩展语料与旧K索引的`blocks.npy` SHA256完全相同。

E10a不直接训练路由器，而是测量：

1. 每query的全head block并集增长和槽位冗余；
2. GQA siblings、同层非siblings和跨层heads的Top-K Jaccard；
3. query-invariant block hub频率；
4. 按dataset分层train/test后，train gold选择的head子集能否在test泛化；
5. 所有head并集、少量稳定heads和多数投票之间的gold recall差异。

480题结果：同层非GQA-sibling Top16 mean Jaccard为0.78%；train选择16 heads的held-out micro/macro recall为53.4%/48.0%，随机16 heads为21.2%/21.5%；但全448 heads Top16并集平均覆盖4,299 blocks，并存在81个对所有queries都出现的universal hubs，其中只有一个在一题上是gold。64题对应结构指标与480题接近。

该结果支持“稳定专业head与公共hub并存”，但E10a本身不支持“已经得到高效head router”。稳定head假设通过随机分层切分，但NarrativeQA/MultiFieldQA样本少，且train选择仍使用gold。

**E10b已完成。** 按dataset分层5折，在四个train folds估计每个`layer x head x block`的全分数均值/方差，只在held-out fold排名。比较raw、减均值和z-score；所有方法共用一次10M扫描，gold只用于事后评估。448 heads的query-invariant prior平均解释49.8%分数方差；z-score把universal hubs从81降到0，把Top16并集召回从71.04%提高到80.63%，把RRF39从22.71%提高到38.13%。同深度配对检验分别为`p=5.91e-6`和`p=1.10e-13`。

**E10c得到探索性结果。** 在每个train fold仅统计raw Top1 block多样性，不使用gold或test queries；选择16 heads后，test z-score Top16并集召回为62.08%，matched random为28.68%。RRF压到39 blocks后为49.79%，高于全448-head z-score的38.13%。但该proxy是在七个候选无标签特征中事后发现，必须冻结到整个dataset或新query外部holdout；不能把当前五折写成确认性验证。

**E10d整数据集留出压力测试已完成。** 依次把六个数据集整体留出；每折的block均值/方差和Top1-diversity head选择都只能读取另外五个数据集。冻结16 heads后，z-score Top16候选并集召回为62.50%，最终RRF39为49.17%，与E10c的62.08%/49.79%基本一致；随机16-head并集期望为28.41%，200个随机重复的经验`p=0.00498`。train-only diversity与held-out单head召回在五个可测数据集上的Spearman平均为0.622，六个训练集上的完整head排序平均Spearman为0.997。该结果排除了同一dataset分层泄漏，但由于proxy最初仍在这480题上发现，只能称为压力测试，不能称为全新外部确认。

**E10e selected-KV连续profile已完成。** 六折20-query-head并集只需要12层、17个`layer x KV-head`通道。无损FP16 profile从143.36 GB压到10.88 GB（7.59%），RRF39保持49.17%，gold命中差异为0。已有packed 1/2卡运行与其他GPU任务重叠，计时作废；正式wall-clock必须在独占GPU上重跑。

**E10f block安全支撑函数界已完成并失败。** 对每个block切1/2/4/8/16段，使用`max_j(q^T c_j + ||q||r_j)`严格上界token级max-QK。在480题、7,680个query-head pairs和全部39,062 blocks上安全违反为0；但16段平均仍保留99.9985% blocks，中位数100%，估算点积速度0.938x。冻结停止条件：不再增加段数，转向有损概率索引加exact精排。

**E10g 查询原型完整block轴已完成。** 480题、7,680个query-head pairs均使用train-only的128个原型。raw目标在10.49%候选下保持95.12% exact Top16；当前最佳检索所需的z-score目标在25%候选下保持90.50%。另行构建30.0 MB的exact-QK train mean/std profile后，25%候选为89.09%，反而略低，排除了“只是代理prior不准”这一解释。当前冻结下一步是直接用候选做exact QK z-score精排并报告最终RRF39 gold recall。

**E10h 原型候选加exact精排已完成。** 对5.24%/10.49%/20.97%/25.00% block预算，最终RRF39 gold recall为41.04%/42.92%/46.04%/46.04%，完整selected-head参考为49.17%。25%预算保留93.64%基线质量，4胜19负；20.97%到25%饱和。质量层面部分通过，但代理粗评分仍线性扫描blocks，因此不能宣称已经实现1B次线性搜索。

Matched BM25-block/record39在同一480题、同一39-block预算下为66.67%/81.04%，显著高于LODO的49.17%。E10b/c/d证明了可利用且跨数据集稳定的内部访问结构，不证明Q/K全局检索优于RAG。完整结果见`query_invariant_prior_and_unsupervised_head_gate_10m_20260714.md`。

### E11：100M与1B多卡系统测试

10M单卡selected-head执行已完成：六个LODO折的20-head并集从741.77秒降到182.72秒，RRF39保持49.17%，实测4.06倍加速；但22.4倍query-head通道缩减只转化为4.06倍墙钟收益。

17个实际使用的`layer x KV-head`已经从原始交错profile预打包为连续文件。等待独占资源后测1/2/4/6/8卡：索引显存、K读取字节、每题wall-clock、跨卡通信、TopK合并和KV加载。若预打包后点积减少仍不能转化为wall-clock收益，需要按带宽/通信/同步分解瓶颈，而不是继续报告理论speedup。完成10M因果profiling后才扩到100M/1B。

## 12. 停止与转向条件

满足以下任一情况，应停止把“局部 residual-K 树”作为主线：

- E7中，在正确seed和正确bridge条件下，KV expansion仍不能提高第二证据或attention token recall；
- E8/E9两个合理改进都只提高centroid邻居代理指标，不提高真实attention或生成；
- 概率查询原型在完整39,062-block轴上无法用小于25%候选保持至少90%的exact Top16；
- 在100M上索引和通信开销抵消候选点积收益；
- 相同预算的RAG二次检索在质量和wall-clock上都稳定更好。

届时应审计目标：centroid几何近邻可能不是下游需要的标签，应把研究目标改为直接预测 query-conditioned attention support，而不是继续优化错误代理指标。
