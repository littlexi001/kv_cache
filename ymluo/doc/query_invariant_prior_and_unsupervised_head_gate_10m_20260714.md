# 10M Context 中的 Query-Invariant Prior 与无标签 Head Gate

> **一句话结论：在9,999,872-token真实语料的480题上，head稀疏保持49.17%召回并实测加速4.06倍；查询原型粗召回加exact精排只读25% blocks时为46.04%，保留基线的93.64%，但代理仍线性扫blocks且matched BM25更强，因此内部KV压缩有效，1B次线性搜索尚未完成。**

**日期：** 2026-07-14  
**模型：** Qwen3-0.6B  
**数据：** 真实LongBench文本组成的10M语料，39,062个256-token blocks  
**状态：** E10b-E10d完成；整数据集留出与真实稀疏执行通过，尚缺完全新queries/新dataset确认

## 1. 研究问题

上一轮全层全head实验发现：

- 28层、每层16个query heads的Top16平均展开为4,299个blocks；
- 81个blocks被全部480个queries提名；
- gold通常只被少数专业heads提名，因此全head多数投票会把它压掉。

本轮检验两个可证伪假设：

```text
H1: score_lh(q,b) = mu_lh(b) + delta_lh(q,b)

H2: 有用head会随query改变Top1 block；
    因而train queries上的Top1-block多样性可以无标签识别专业head。
```

`mu_lh(b)`是某个layer/head对block的固定吸引力；`delta_lh(q,b)`是当前query相关残差。H1若成立，跨折去均值或z-score应减少hubs并提高held-out召回。H2若成立，head gate不需要gold或文本RAG特征。

## 2. 冻结实验协议

### 2.1 数据与真实Q/K

| 项目 | 数值 |
|---|---:|
| 语料tokens | 9,999,872 |
| blocks | 39,062 |
| block size | 256 tokens |
| queries | 480 |
| 数据集 | 2WikiMQA 136、HotpotQA 135、MuSiQue 135、Qasper 59、MultiFieldQA 13、NarrativeQA 2 |
| 模型通道 | 28层 x 16 query heads = 448 heads |
| KV heads | 每层8个 |
| 表征 | 真实pre-RoPE Q/K投影到冻结SVD32空间 |
| 每head保存 | Top16 blocks |

没有合成高斯向量。K来自10M文本真实前向；Q来自自然语言问题。gold只用于运行后的block recall评估。

### 2.2 五折无泄漏校准

480题按dataset分层、固定seed分为5折，每折94到98题。对held-out折`f`：

```text
mu_lh,-f(b)    = mean_{q not in f} score_lh(q,b)
sigma_lh,-f(b) = std_{q not in f}  score_lh(q,b)

raw      = score_lh(q,b)
centered = score_lh(q,b) - mu_lh,-f(b)
zscore   = centered / max(sigma_lh,-f(b), 1e-4)
```

校准不读取test query，也不使用任何gold。所有方法在同一次10M矩阵扫描中计算。Top-K显式按“score降序、同分block ID升序”排序，避免GPU数量改变tie-breaking。

### 2.3 整数据集留出压力测试

Top1-block多样性在五折比较七个proxy后被选中，因此五折结果是探索性发现。进一步做6折leave-one-dataset-out（LODO）：每次把2WikiMQA、HotpotQA、MultiFieldQA、MuSiQue、NarrativeQA或Qasper中的一个数据集全部留出。

- `mu/sigma`只由另外五个数据集估计；
- head diversity和head选择也只读取另外五个数据集；
- 留出数据集的query不参与校准或gate；
- gold仍只用于最后评估。

这比dataset-stratified五折更严格，但还不是完全外部确认，因为Top1-diversity这个proxy最初仍是在同一批480题上发现的。

### 2.4 方差分解

每个layer/head在全部queries和blocks上的分数满足全方差分解：

```text
Var_{q,b}[s]
  = Var_b(E_q[s | b]) + E_b(Var_q[s | b])

prior_fraction
  = Var_b(E_q[s | b]) / Var_{q,b}[s]
```

第一项是固定block prior，第二项是query-dependent变化。这个指标只描述结构，不直接等于检索质量。

## 3. H1结果：固定prior真实存在且可以分离

### 3.1 方差量化

448个heads的`prior_fraction`：

| 统计量 | 数值 |
|---|---:|
| mean | 49.80% |
| median | 50.50% |
| p05 | 17.56% |
| p95 | 80.26% |
| min / max | 3.13% / 93.55% |

平均约一半QK分数方差来自query-invariant block差异。层间并不一致：layer 1均值29.68%，layer 15为66.86%。因此不能对所有layers/heads使用同一个校准强度。

数值恒等式`total = between + within`的平均相对误差为`1.15e-5`，排除了明显的流式统计错误。

### 3.2 Hub与召回

以下结果均为每head Top16，最终共识仍只返回39个blocks：

| 方法 | 平均不同候选blocks | Gold并集召回 | RRF39召回 | 至少覆盖半数queries的hubs | Universal hubs |
|---|---:|---:|---:|---:|---:|
| raw | 4,299.3 | 71.04% | 22.71% | 2,322 | 81 |
| centered | 4,805.0 | 77.08% | 27.50% | 333 | 0 |
| z-score | 5,152.4 | **80.63%** | **38.13%** | **266** | **0** |

z-score相对raw：

- Top16并集召回`+9.58pp`，74题救回、28题损失，McNemar `p=5.91e-6`；
- RRF39召回`+15.42pp`，90题救回、16题损失，`p=1.10e-13`；
- universal hubs从81降为0。

这支持H1：公共hubs至少部分来自可估计的静态prior，而不是当前query相关性。

### 3.3 不是靠无限扩大候选

跨预算比较更严格：

| 方法 | 每head深度 | Head slots | 平均不同候选 | Gold并集召回 | RRF39召回 |
|---|---:|---:|---:|---:|---:|
| raw | 16 | 7,168 | 4,299 | 71.04% | 22.71% |
| z-score | 8 | 3,584 | **2,813** | 74.17% | **39.17%** |

z-score Top8的并集提升`+3.13pp`尚不显著，`p=0.155`；但固定39-block输出提升`+16.46pp`，92题救回、13题损失，`p=7.98e-16`。因此主要收益不是多读blocks，而是让有限共识预算更容易保留相关证据。

## 4. H2结果：Top1-block多样性识别query-responsive heads

### 4.1 无标签gate

对每个test折，仅在另外四折计算：

```text
diversity_lh
  = number of distinct raw Top1 blocks on train queries
    / number of train queries
```

然后选择多样性最高的heads，并在test上读取对应的cross-fitted z-score排名。gate不看问题文本、test query、gold或BM25/E5特征。

七个候选proxy全部被保留在结果文件中。Top1-block多样性是看过本轮结果后发现的winner，因此属于探索性发现，必须做外部holdout。

### 4.2 五折召回与稳定性

| Heads | 无标签gate test recall | Dataset-macro | Matched random | Random p95 | 经验p值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.63% | 28.64% | 3.14% | 8.54% | 0.00498 |
| 4 | 47.29% | 44.51% | 10.24% | 17.50% | 0.00498 |
| 16 | **62.08%** | **63.54%** | 28.68% | 35.84% | 0.00498 |
| 64 | 68.13% | 67.91% | 53.07% | 57.50% | 0.00498 |

五折中：

- Top1始终是唯一的`L3/H10`；
- Top4集合完全一致；
- Top16集合平均fold-pair Jaccard为0.906。

Top1 score方差、Top-K score方差和raw score gap反而不如随机。有效信号不是“分数波动大”，而是“排名第一的block随query稳定地改变”。这是order-statistic层面的query responsiveness。

### 4.3 从10M候选到固定读取预算

| Heads | 每head深度 | 平均不同候选blocks | 候选对应tokens | 并集召回 | RRF39召回 |
|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 59.0 | 15.1K | 47.29% | 43.33% |
| 16 | 8 | 106.0 | 27.1K | 55.21% | 47.50% |
| 16 | 16 | 209.6 | 53.6K | **62.08%** | **49.79%** |
| 64 | 16 | 638.0 | 163.3K | 68.13% | 44.58% |

16-head Top16是当前内部检索的最佳固定39-block结果。最终只需加载`39 x 256 = 9,984 tokens`，占10M全文约0.10%。64 heads虽提高并集召回，却使RRF39下降，直接证明不相关heads会污染共识。

原五折结果只实测了质量；完整GPU扫描仍计算全部448 heads。第4.5节进一步给出真实selected-head执行，不能再用28倍算术量代替实测。

### 4.4 整数据集留出结果

冻结Top1-diversity后，在每个LODO折只用另外五个数据集选择heads：

| 协议 | 16-head候选blocks | 候选tokens | 并集召回 | RRF39召回 |
|---|---:|---:|---:|---:|
| 原dataset-stratified五折 | 209.6 | 53,646 | 62.08% | 49.79% |
| 整数据集留出LODO | 210.3 | 53,847 | **62.50%** | **49.17%** |

LODO最终召回只比原五折低0.62pp。LODO中，16-head并集召回相对随机16-head期望为`62.50% vs 28.41%`，超过200个随机重复的最大观测范围，经验`p=0.00498`；六折head集合的平均Jaccard为0.814。

按留出数据集拆分的16-head候选召回为：2WikiMQA 71.32%、HotpotQA 74.81%、MultiFieldQA 84.62%、MuSiQue 38.52%、NarrativeQA 50.00%（仅2题）、Qasper 64.41%。性质跨数据集成立，但MuSiQue仍是主要弱项。

机制证据也跨数据集：在五个有非零head召回差异的数据集上，train-only Top1 diversity与held-out单head Top16召回的Spearman平均为0.622；完整448-head diversity排序在六个LODO训练集之间的平均Spearman为0.997。也就是说，`query responsiveness`更像稳定head属性，而不是某一折的偶然排序。

### 4.5 真实selected-head稀疏执行

六个LODO折各自选择16 heads；为了在一次480题批处理中同时覆盖六折，实际扫描其并集20/448个query-head channels，分布在12/28层和17/224个`layer x KV-head` channels。raw、centered、z-score仍在同一次扫描中计算。

| 实现 | GPU | 实际扫描channels | 时间 | RRF39召回 |
|---|---:|---:|---:|---:|
| 完整扫描 | 1 | 448 query heads，28层 | 741.77 s | 49.17% |
| selected-head稀疏扫描 | 1 | 20 query heads，12层 | **182.72 s** | **49.17%** |

实测加速为`741.77 / 182.72 = 4.06x`，墙钟减少75.37%。稀疏与完整扫描的480题gold命中零差异；475题的最终RRF39顺序完全一致，478题的39-block集合完全一致，平均集合重合率为99.989%。少数排序差异来自不同einsum形状产生的最大0.00126分数误差。

20/448是22.4倍query-head通道缩减，但wall-clock只有4.06倍。这说明剩余瓶颈主要是原始交错K-profile读取、逐层启动、block遍历和Top-K维护。当前稀疏扫描仍从原始全KV profile做`np.take`，因此4.06倍不是head稀疏性的最终系统上限。

### 4.6 无损连续profile与计时审计

已把六折并集实际需要的12层、17个`layer x KV-head`通道打包为连续FP16 profile。原始全profile为143.36 GB，packed profile为10.88 GB，只占7.59%；打包不依赖gold，480题RRF39仍为49.17%，gold命中差异为0。

已有packed 1卡/2卡运行与其他LongBench任务重叠，GPU显存和计算资源不独占，因此其时间不能作为正式加速结论。当前唯一保留的正式selected-head时间仍是无GPU争用时的182.72秒和相对完整单卡的4.06倍；packed profile必须在独占1/2/4/8卡上重跑。

## 5. 与matched BM25/RAG的边界

同一10M语料、同一480题、同一39-block输出预算：

| 方法 | Gold block recall@39 |
|---|---:|
| 无标签16-head z-score + RRF，原五折 | 49.79% |
| 无标签16-head z-score + RRF，LODO | 49.17% |
| BM25 block | 66.67% |
| BM25 record20 | 77.08% |
| BM25 record30 | 80.63% |
| BM25 record39 | **81.04%** |

严格LODO KV方法相对BM25-block低17.50pp，22题KV独有命中、106题BM25独有命中，McNemar `p=2.21e-14`。因此不能声称当前Q/K搜索优于RAG。

Q/K仍有补充信号：

- LODO相对BM25-record39仍有28题KV独有命中；二者oracle union为86.88%；
- Qasper上KV为44.07%，BM25-block/record39为40.68%/32.20%；
- LODO固定等权RRF只把BM25-block从66.67%提高到67.08%，16胜14负，`p=0.856`；
- 与更强record30/39融合反而下降，说明无条件融合失败。

当前边界是：RAG擅长共享文本语义/词法定位；Q/K属性描述模型内部、layer/head-specific、可随生成状态变化的访问模式。后者尚未在RAG无法文本化的动态状态或attention目标上形成已验证优势。

## 6. 对“为什么1B可以高效搜索”的理论更新

本轮支持的结构不是“1B tokens天然可以一次Top-K”，而是QK score tensor存在两种可压缩轴：

```text
block axis:
  大量query-invariant prior可以离线估计和校准

head axis:
  少数query-responsive heads跨train/test稳定；
  大量heads主要产生公共hub或投票噪声
```

更准确的候选模型是：

```text
score_lh(q,b)
  = mu_lh(b)
  + a_lh * r_lh(q,b)
  + epsilon_lh(q,b)

active_heads
  = heads with stable rank responsiveness across calibration queries
```

`prior_fraction`与z-score收益只有弱Spearman相关`rho=0.189`；最高prior四分位并非收益最大。说明仅有大`mu`不够，head还必须保留可区分的query residual。Top1-block多样性直接测量了这个残差是否能改变极值排名。

### 6.1 1B存储与计算含义

1B tokens按256切块约为3,906,250 blocks：

- 全448 heads保存FP16均值+标准差约7.0 GB；
- 只对16个稳定heads保存约250 MB；
- head channel算术量理论上从448降到16，即28倍减少。

但block轴仍需搜索约390万个blocks。本轮只解决了head轴和hub校准，没有得到sublinear block index，也没有测1B K paging、带宽或通信。因此它是1B高效搜索的一个必要结构，不是完整答案。

### 6.2 Block支撑函数与查询方向结构

selected-head原始分数可写成block的支持函数：

```text
h_b(q) = max_t q^T k_bt
score(q,b) = mean_i h_b(q_i)
```

若把每个block切成若干段，保存段中心`c_bj`和覆盖半径`r_bj`，则有严格安全上界：

```text
h_b(q) <= max_j [q^T c_bj + ||q|| r_bj]
```

在480题、7,680个有效query-head pair、全部39,062 blocks上，1/2/4/8/16段均为0次安全性违反。但16段仍平均保留99.9985%的blocks，中位数保留100%，估算点积速度仅0.938x。也就是说，这个界数学上正确，但极值几何使覆盖半径过大，实际上不能剪枝。

查询方向本身存在中等强度的低维结构。严格LODO的96个fold-head模型中，32维方向的平均rank90为18.10、有效秩为11.62；但held-out方向对rank16训练子空间的平均残差仍有0.366。128个train-only球面原型对held-out query token的最近余弦均值约0.80，覆盖并不紧。

使用最近查询原型的block支撑值作为有损代理，在143个held-out queries、2,288个query-head pairs、每对均匀抽样512个真实blocks上，代理分数与exact分数的平均Spearman为0.819；取16/32/64/128个代理候选时，对抽样exact Top16的平均覆盖率为45.05%/64.80%/82.15%/93.77%。但Top1一致率只有19.62%，严格Lipschitz上界仍保留99.9997%候选。这一结果只支持“概率粗路由+exact精排”，不支持零损失安全剪枝，也不是完整10M gold recall。

完整39,062-block轴、全部480题和7,680个query-head pairs确认raw结构很强：2,048/4,096/9,766候选分别覆盖89.51%/95.12%/98.80%的raw exact Top16。对最终系统使用的z-score exact Top16，相同候选只覆盖63.94%/76.89%/90.50%。另行用真实QK构建96个fold-head的train-only exact mean/std profile（30.0 MB、151.1秒）后，9,766候选覆盖反而为89.09%，没有修复损失。

因此损失主要来自标准化对方向代理残差的放大，而不只是prior估计不准：低方差block除以小标准差后，原本不大的raw支持函数误差会改变极值排序；代理自己的mean/std产生了一定误差抵消。当前属性支持raw概率路由，但对z-score路由只能在25%候选处勉强达到90%。这些指标仍是per-head exact Top16保持率，不是gold block recall。

最终把代理候选交给真实QK和train-only exact prior做z-score精排，再对16 heads做RRF39：5.24%/10.49%/20.97%/25.00% block预算的gold recall分别为41.04%/42.92%/46.04%/46.04%，完整selected-head参考为49.17%。25%预算保留了`46.04 / 49.17 = 93.64%`的基线召回，净损失3.13pp；配对上4题新命中、19题丢失。20.97%到25%没有继续改善，说明主要瓶颈已经是少量高价值head vote未进入代理候选，而不是单纯继续扩大exact精排预算。

## 7. 运行时间与复现审计

| 阶段 | 实测 |
|---|---:|
| 1 x RTX 3090，480题、448 heads、39,062 blocks，LODO raw+centered+z-score同扫 | 741.77 s |
| 1 x RTX 3090，480题、LODO六折并集20 heads、12层，三路同扫 | **182.72 s** |
| 4 x RTX 3090，480题、448 heads、39,062 blocks，raw+centered+z-score同扫 | 357.33 s |
| 连续selected-KV profile存储 | 10.88 GB，为全profile的7.59% |
| 平均每层 | 12.76 s |
| BM25 block建索引并打分480题 | 11.62 s，CPU批量 |
| BM25 record建索引并打分480题 | 11.25 s，CPU批量 |
| BM25最终480题排序 | 3.71 s |

同一全head扫描从1卡到4卡只有`741.77 / 357.33 = 2.08x`实测加速，远低于理想4x；相比之下，减少无用层/head的单卡稀疏执行比完整4卡还快1.96倍。单卡运行时GPU独占，但服务器同时有其他任务产生共享CPU/I/O压力，因此这些数字是当前系统实测，不是硬件峰值。QK时间是离线批量扫描，不是单请求延迟；BM25口径也不是端到端生成时间。

连续profile已有的1卡和2卡时间因与其他GPU任务重叠而作废，未列入上表，也不用于计算加速比。这是资源审计后的主动撤回，不是缺失结果。

raw score与旧7卡冻结扫描逐元素完全相同，最大绝对误差为0。约3.87%的ID槽位因旧实现的同分隐式tie顺序不同；新实现统一使用block ID二级键，所有差异位置的rank score完全相等。

## 8. 失败、限制与下一步

### 已支持

1. query-invariant `head x block` prior真实存在，平均解释约一半分数方差。
2. cross-fitted z-score显著减少hubs并提高固定39-block共识召回。
3. train-only Top1-block多样性可以无标签识别稳定query-responsive heads。
4. 少量heads比全heads共识更好，支持head-specific组合稀疏性。
5. 整数据集LODO下16-head RRF39为49.17%，与原五折49.79%基本一致，说明性质不是dataset混合泄漏造成的。
6. 真实20-head并集稀疏扫描保持49.17%召回并取得4.06倍单卡加速，证明head属性已转化为实际执行收益。
7. 17个`layer x KV-head`通道可无损打包到全profile的7.59%，说明head稀疏性同时压缩存储轴。
8. 查询方向具有中等低秩结构；完整block轴上10.49%原型候选保持95.12% raw exact Top16，支持概率粗路由。

### 仍失败或不足

1. matched BM25显著强于当前Q/K方法；等权融合没有显著收益。
2. Top1-diversity已通过整数据集留出压力测试，但它最初在同一480题的七个proxy中胜出，仍缺完全新queries/新dataset确认。
3. 只测gold block recall，没有测post-RoPE exact attention、Value、生成答案或PPL。
4. prior是corpus-specific；新文档、新模型或新领域是否需要重新校准未知。
5. 本轮仍线性扫描全部blocks，未解决1B block轴的次线性检索。
6. NarrativeQA只有2题，macro结果不能视为该任务的稳定估计。
7. center-radius与query-prototype Lipschitz严格上界都几乎保留100% blocks；当前几何界不能解决1B次线性精确搜索。
8. z-score目标明显削弱原型路由：25%候选只保持90.50% per-head Top16；exact prior也不能修复，并且这些都不是gold召回。
9. 当前原型代理评分仍线性扫描全部block支撑值；候选比例下降不等于已经获得次线性query time。

### 冻结下一步

1. **外部holdout：** 冻结`z-score + Top1-diversity 16 heads + Top16 + RRF39`，在未参与feature比较的新queries/新dataset上复验。
2. **预打包与多卡：** 连续profile已经构建；等待独占GPU后重测1/2/4/8卡wall-clock、K读取量和Top-K通信，作废所有资源争用计时。
3. **attention对齐：** 对39-block结果做真实位置post-RoPE全维QK/Value精排，测attention mass与生成质量。
4. **RAG-miss实验：** 冻结BM25/RAG，在其miss集合上训练无泄漏task/state gate，检验Q/K独有29题能否被可预测地救回。
5. **block轴结构：** exact精排最终召回已经完成；下一步保护高价值head votes或学习z-score-aware代理，并把线性支撑表扫描替换为原型倒排Top-list或ANN。
6. **100M/1B：** 只有外部holdout和selected-head attention实验通过后，才构建真实大规模索引。

## 9. 产物

- 分布式去偏扫描器：`ymluo/projects/parallel_block_retrieval/src/run_all_head_prior_debiased_retrieval.py`
- 去偏汇总器：`ymluo/projects/parallel_block_retrieval/src/summarize_head_prior_debiasing.py`
- 无标签head gate：`ymluo/projects/parallel_block_retrieval/src/analyze_unsupervised_head_gate.py`
- matched BM25配对：`ymluo/projects/parallel_block_retrieval/src/compare_head_gate_to_bm25.py`
- 完整运行脚本：`ymluo/projects/parallel_block_retrieval/scripts/run_head_prior_debiasing_10m_480q_20260714.sh`
- LODO运行脚本：`ymluo/projects/parallel_block_retrieval/scripts/run_head_prior_debiasing_10m_dataset_lodo_20260714.sh`
- LODO机制分析：`ymluo/projects/parallel_block_retrieval/src/analyze_dataset_lodo_head_responsiveness.py`
- selected-head扫描器：`ymluo/projects/parallel_block_retrieval/src/benchmark_selected_head_debiased_retrieval.py`
- selected-head验证器：`ymluo/projects/parallel_block_retrieval/src/verify_selected_head_retrieval.py`
- selected-KV连续profile：`ymluo/projects/parallel_block_retrieval/src/pack_selected_kv_profile.py`
- 安全支撑函数界：`ymluo/projects/parallel_block_retrieval/src/evaluate_selected_kv_support_bounds.py`
- 查询方向流形：`ymluo/projects/parallel_block_retrieval/src/analyze_lodo_query_manifold.py`
- 查询原型代理：`ymluo/projects/parallel_block_retrieval/src/evaluate_query_prototype_support_bound_sample.py`
- 完整block轴原型代理：`ymluo/projects/parallel_block_retrieval/src/evaluate_query_prototype_full_axis.py`
- exact train-only prior：`ymluo/projects/parallel_block_retrieval/src/build_selected_head_exact_prior_profile.py`
- 原型候选exact精排：`ymluo/projects/parallel_block_retrieval/src/evaluate_prototype_exact_rerank_lodo.py`
- 主结果：`ymluo/projects/parallel_block_retrieval/outputs/head_prior_debiasing_10m_query480_20260714_v1/`
- LODO结果：`ymluo/projects/parallel_block_retrieval/outputs/head_prior_debiasing_10m_dataset_lodo_query480_20260714_v1/`
- selected-head结果：`ymluo/projects/parallel_block_retrieval/outputs/selected_head_lodo_scan_10m_query480_20260714_v1/`
- matched BM25：`ymluo/projects/parallel_block_retrieval/outputs/real_longbench_docqa_10m_record480_bm25_20260714_v1/`
- block轴与查询流形证据：`ymluo/doc/1b_context_search_research_exploration/evidence/block_axis_support_and_query_manifold_10m_20260714.json`
