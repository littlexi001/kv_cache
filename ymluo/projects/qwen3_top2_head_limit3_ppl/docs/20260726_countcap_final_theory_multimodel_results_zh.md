# Direct CountCap：理论与双模型正式结果

更新时间：2026-07-26

## 1. 冻结方法

当前方法在首个 2048-token prefill chunk 上建立 sampled uncentered PCA48，随后固定；再使用 grouped log-scale INT4 Key + INT8 Query + 256 点 sampled-quantile + direct sparse attention。它不使用训练式 router、任务标签、exact-QK 重排或 Full 回退。

目标预算为：

$$
B(N)=\min\left(N,1280,\max\left(256,\left\lceil0.06N\right\rceil\right)\right).
$$

当 $N\le256$ 时，$B(N)=N$，此时精确消费全部可用历史。这是预算公式的饱和边界，不是风险、任务或成本触发的 Full fallback。

`1280` 是分位数目标，不是当前 kernel 的绝对 hard cap；正式方法说明见 `docs/20260726_direct_countcap_frozen_method_reproduction_zh.md`。

## 2. LongBench 16 任务

| 模型 | Full Macro | CountCap Macro | 质量保持率 | 95% CI（Macro 差值） | 平均目标 token/head | Decode ms/token speed | Online ms/token speed | 整样本 Total latency speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.4456 | 0.4461 | 100.11% | [-0.0023, 0.0032] | 382.5 | 0.914x | 0.896x | 0.825x |
| Qwen3-4B-Instruct | 0.4099 | 0.4105 | 100.15% | [-0.0029, 0.0044] | 383.8 | 0.770x | 0.777x | 0.744x |

`Decode/Online speed` 按各方法实际生成 token 数归一化；`Total latency speed` 是整条样本延迟比，会受输出长度差异影响，只作为应用延迟补充，不作为纯 decode kernel 加速结论。

### sampled-quantile 实际消费审计

| 模型 | 样本 | 平均 prompt | 目标比例 | 实际比例 | 样本内 p95 比例均值 | 最大比例 | 实际 token/head 均值 | p95 token 均值 | 最大 token | capacity overflow head 比例 | host-side proxy top-k fallback 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 64 | 6476.5 | 7.12% | 7.26% | 9.44% | 25.16% | 420.8 | 558.6 | 1036 | 0.02% | 0.00% |
| Qwen3-4B-Instruct | 64 | 6512.0 | 7.06% | 7.26% | 9.55% | 24.49% | 427.2 | 572.4 | 1044 | 0.03% | 0.00% |

实际消费来自开启诊断的独立 m4 审计，不使用目标预算代替执行数量。该诊断会增加开销，因此其时间不用于速度主张。
冻结的 qprojscan/qkvfused 是异步 sampled-quantile 路径：极少数超过 capacity 的 head 由 CUDA kernel 在 capacity 处截断，不触发host-side proxy top-k。表中的 fallback 为 0；即使非异步变体触发该字段，它也只表示完整低比特 proxy top-k，不是 Full Attention。

7.5K 左右的短上下文结果用于验证质量，不构成长序列加速主张。该 runner 保留完整 GPU FP16 K/V。

### 64K/128K 冻结方法配对测速

| 模型 | 历史长度 | 配对 case | Full PPL | CountCap PPL | PPL 保持率 | 实际 token/head | per-head 范围 | 实际比例 | Full ms/token | CountCap ms/token | $\Delta T_{fixed}$ (s) | break-even token | Decode speed | Prefill+decode speed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 64000 | 4 | 11.4464 | 11.7336 | 97.55% | 1560.8 | 86-3851 | 2.44% | 148.52 | 56.18 | -0.028 | 0 | 2.643x | 1.316x |
| llama31_8b | 128000 | 4 | 11.7955 | 12.6481 | 93.26% | 1500.4 | 13-6411 | 1.17% | 272.86 | 59.45 | -0.014 | 0 | 4.590x | 1.240x |
| qwen3_4b | 64000 | 4 | 18.0739 | 19.7815 | 91.37% | 1467.2 | 116-3851 | 2.29% | 158.53 | 57.18 | +0.591 | 6 | 2.772x | 1.395x |
| qwen3_4b | 128000 | 4 | 17.9150 | 20.7108 | 86.50% | 1440.0 | 15-6410 | 1.12% | 297.54 | 72.26 | +0.374 | 2 | 4.118x | 1.261x |

该表在相同文本窗口、相同预测 token 数下配对；`Decode speed` 包含 PCA/INT4 检索、sampled threshold、候选 gather 和精确稀疏 attention。`Prefill+decode speed` 还计入 dense prefill 与索引构建。

令 $F$ 为每个请求的 prefill/索引固定成本，$t$ 为每个 decode step 的在线成本，则摊销交叉点按配对计时定义为

$$
n^*=\max\left(0,\frac{F_{\mathrm{CountCap}}-F_{\mathrm{Full}}}{t_{\mathrm{Full}}-t_{\mathrm{CountCap}}}\right).
$$

最差长文本 PPL 保持率为 86.50%（qwen3_4b，128K）。这构成“PCA/量化尾部扰动对任意自然文本都可忽略”的反例；LongBench 任务质量与通用连续文本 PPL 必须分别陈述。

### 分任务结果

| 任务 | Llama Full | Llama CountCap | 保持率 | Qwen Full | Qwen CountCap | 保持率 |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimqa | 0.4581 | 0.4614 | 100.73% | 0.3962 | 0.4040 | 101.97% |
| gov_report | 0.3320 | 0.3387 | 102.04% | 0.3020 | 0.3064 | 101.46% |
| hotpotqa | 0.4904 | 0.4904 | 100.00% | 0.4358 | 0.4472 | 102.63% |
| lcc | 0.6305 | 0.6264 | 99.35% | 0.6472 | 0.6424 | 99.26% |
| multi_news | 0.2681 | 0.2675 | 99.81% | 0.2405 | 0.2407 | 100.10% |
| multifieldqa_en | 0.5730 | 0.5737 | 100.11% | 0.4952 | 0.4881 | 98.55% |
| musique | 0.2567 | 0.2644 | 102.99% | 0.1321 | 0.1365 | 103.38% |
| narrativeqa | 0.2277 | 0.2284 | 100.33% | 0.1781 | 0.1833 | 102.92% |
| passage_count | 0.0740 | 0.0758 | 102.48% | 0.0067 | 0.0000 | 0.00% |
| passage_retrieval_en | 0.6200 | 0.6200 | 100.00% | 0.6000 | 0.6000 | 100.00% |
| qasper | 0.4533 | 0.4265 | 94.09% | 0.4210 | 0.4126 | 98.02% |
| qmsum | 0.2277 | 0.2323 | 102.05% | 0.2129 | 0.2082 | 97.79% |
| repobench-p | 0.5045 | 0.5147 | 102.02% | 0.5057 | 0.5169 | 102.21% |
| samsum | 0.4457 | 0.4493 | 100.81% | 0.4443 | 0.4414 | 99.35% |
| trec | 0.6700 | 0.6700 | 100.00% | 0.6800 | 0.6800 | 100.00% |
| triviaqa | 0.8977 | 0.8977 | 100.00% | 0.8605 | 0.8605 | 99.99% |

### 与 RaBitQCache 公开结果的协议级参考

| 来源 | Full（×100） | 稀疏方法 | 稀疏分数（×100） | 平均目标/预算 |
|---|---:|---|---:|---:|
| 本次同 runner 的 13-task 子集 | 45.69 | CountCap | 45.71 | 7.60% target |

| RaBitQCache 论文中的方法 | 设置 | 分数（×100） | 实际预算 |
|---|---|---:|---:|
| Full | - | 50.58 | 100.00% |
| RaBitQCache | top-p=0.95 | 50.63 | 17.33% |
| Quest | 1024 | 46.52 | 11.38% |
| Double Sparsity | 1024 | 50.28 | 11.42% |
| SparQ | ratio=0.25 | 50.15 | 25.00% |
| MagicPIG | official default | 49.95 | - |
| PyramidKV | official default | 45.09 | - |
| SnapKV | official default | 44.91 | - |
| PQCache | official default | 50.34 | - |
| KIVI | official default | 50.13 | - |

RaBitQCache 行是论文公开数字，不是本仓库同环境复现。其前两层使用 Full，硬件、推理框架、prompt 与 stop policy 也不同，只能比较相对保持率和预算量级。

### 与 Self-Indexing KVCache 公开结果的协议级参考

| 来源 | Full（×100） | 稀疏方法 | 稀疏分数（×100） | 质量保持率 | 预算 |
|---|---:|---|---:|---:|---:|
| 本次同 runner 的 11-task 子集 | 53.25 | CountCap | 53.20 | 99.91% | 375.2 target token/head |
| Self-Indexing 论文 | 58.70 | Self-Indexing 16-bit | 58.40 | 99.49% | 160 token |
| Self-Indexing 论文 | 58.70 | Self-Indexing 2-bit | 58.20 | 99.15% | 160 token |

Self-Indexing 的 160-token 预算包含 64 个固定 full-precision sink 和 96 个动态 token，并同时把 K/V 压到低比特；CountCap 不保留 sink 特判且完整 FP16 K/V 仍驻留 GPU。两者任务子集相同，但 runner、prompt、stop policy 与存储目标不同，所以只比较各自相对 Full 的保持率，不直接比较绝对分数。

## 3. Llama 同环境基线

| 方法 | Macro | 相对 Full | 预算 |
|---|---:|---:|---:|
| FullAttention | 0.3658 | 100.00% | 100% |
| SnapKV | 0.3644 | 99.62% | 1024 |
| AdaKV | 0.3636 | 99.39% | 1024 |
| HybridRecentRAG | 0.3405 | 93.08% | - |
| H2O | 0.3348 | 91.53% | 1024 |

## 4. QK 候选误差链

| 指标 | 32K、4% 候选 |
|---|---:|
| 普通 exact top-k recall | 47.34% |
| attention-mass weighted recall | 92.38% |
| 保留 full-attention mass | 86.47% |
| 相对 Exact-QK top-k 的 mass regret | 4.98% |
| 确定性 mass 下界通过率 | 100.00% |
| attention 输出界通过率 | 100.00% |

集合 recall 不是最终质量的充分统计量。真正与输出误差直接相连的是遗漏 attention mass，以及候选内外 Value 条件均值的差。

真实 256 点 midpoint sampled-quantile 还会引入阈值估计误差。下表中的候选集未做人为截断，用于把阈值误差与 capacity 截断分开：

| sampled-quantile 指标 | 32K、目标 4% |
|---|---:|
| 实际选中比例均值 | 3.98% |
| 实际选中比例 p90 | 5.17% |
| 超过 production capacity 的比例 | 0.00% |
| 相对精确 proxy 分位点的绝对 score 误差 | 0.18141 |
| 未截断候选保留的 full-attention mass | 86.22% |

固定 256 点分位数采样在极低目标比例下不是等相对精度的。若暂用独立同分布连续分数近似，样本数为 $m$、目标比例为 $f$，则

$$
\operatorname{Std}(\widehat f)\approx\sqrt{\frac{f(1-f)}{m+2}},\qquad\frac{\operatorname{Std}(\widehat B)}{B}\approx\sqrt{\frac{1-f}{(m+2)f}}.
$$

取 $m=256$ 时，32K/4%、64K/2%、128K/1% 的预算相对标准差近似为 30.5%、43.6%、61.9%。生产采样点是确定性均匀位置且彼此相关，所以这不是严格置信区间；它用于解释为什么长度增大、目标比例降低后，固定采样数会带来更强的 per-head 候选数量抖动。该误差必须与 PCA/SVD 子空间误差、INT4 量化误差和 capacity 截断分别统计。

对确定性 midpoint 采样也不存在分布无关保证。若超过当前阈值的位置只形成 $C$ 个连续区间，则分层 midpoint tail-fraction 误差可由约 $2C/m$ 控制；若高分位置高度碎片化，$C$ 随长度增长，该界即失效。极端情况下可把高分全部放在未采样位置。因而真实候选数量与保留 attention mass 是必要指标，不能用 256 个固定样本替代验证。

## 5. 双模型中心化 QK 奇异谱

softmax 对每个 query 的分数整体平移不敏感。因此令 $H=I-\mathbf1\mathbf1^T/N$，以下主要分析 $S_c=SH=Q(HK)^T/\sqrt d$，而不是可能被无效均值模态主导的原始 $S$。

| 模型 | 原始 rank-1 能量 | 行中心化删除的无效能量 | 第一右奇异向量与常数方向对齐 | 中心化 rank-1 能量 |
|---|---:|---:|---:|---:|
| llama31_8b | 97.19% | 88.83% | 91.13% | 70.89% |
| qwen3_4b | 82.44% | 62.70% | 70.60% | 67.18% |

| 模型 | K 有效秩 | 原始 QK 有效秩 | 中心化 QK 有效秩 | 中心化 rank-16 | rank-32 | 最优 rank-48 | 未中心化 Key-PCA48 | 最优性差距 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama31_8b | 24.67 | 1.22 | 4.72 | 94.79% | 98.01% | 99.09% | 91.10% | 7.99% |
| qwen3_4b | 14.91 | 2.79 | 5.43 | 93.50% | 97.08% | 98.52% | 91.45% | 7.07% |

原始 QK 的极低有效秩部分来自 Key 均值造成的逐行常数，这个分量会被 softmax 精确抵消。中心化后有效秩仍远低于 128，且最优 rank-48 仍保留接近全部分数能量，说明低秩现象不是均值伪影；但 Key-only PCA 与最优 QK-SVD 之间仍有可测差距。

对同一个 Key 矩阵，若 $K=U\Sigma V^T$，则 $K^TK=V\Sigma^2V^T$。因此 sampled uncentered PCA 与同样本 right-SVD 数学上完全等价；旧实验中两者的主子空间重合度为 1.0000，检索指标只存在约 $10^{-5}$ 的数值差异。直接分解中心化 QK 则同时利用了真实 Query，只能作为更强的解释上界。

对任意正交 Key 投影 $P$，本文按残差定义 score fidelity：

$$
\mathcal F_r(P)=1-\frac{\|Q(I-P)\widetilde K^T\|_F^2}{d\|S_c\|_F^2}.
$$

不能一般性地把 $\|QP\widetilde K^T\|_F^2/d$ 直接叫作保留能量，因为保留项与残差项未必 Frobenius 正交。最优 QK-SVD 可写成 Key 协方差白化坐标中的 query-aware 斜投影；它通常不是当前 Key-PCA 使用的正交投影。表中“最优性差距”严格等于实际残差超过 Eckart--Young 最优 rank-$r$ 尾谱残差的归一化部分。

对应的精确三项分解是：

$$
K=\mathbf1\mu^T+\widetilde KP+\widetilde K(I-P).
$$

第一项产生逐 query 常数分数，softmax 精确忽略；第二项由低维索引保留；只有第三项是真正影响排序的谱尾误差。INT4/INT8 是投影坐标中的额外数值扰动。谱尾不能被称为“没有语义”，其可忽略性必须由边界 attention mass 与下游 logit 稳定共同验证。

### 理想 full-history basis 与真实 first-2K basis

| 模型 | Full uncentered-Key PCA48 中心化 fidelity | Full centered-Key PCA48 | Sampled-full 中心化 fidelity | 真实 first-2K 中心化 fidelity | First-2K 中心化 cosine | First-2K/full 子空间 overlap |
|---|---:|---:|---:|---:|---:|---:|
| llama31_8b | 91.10% | 88.59% | 90.27% | 62.39% | 0.7953 | 59.70% |
| qwen3_4b | 91.45% | 91.50% | 89.76% | 59.07% | 0.7609 | 57.23% |

| 模型 | 原始 Key/Query 交换子 | 中心化 Key/Query 交换子 | 全历史 stride-32/full Key 子空间 overlap | 首 2K/full Key 子空间 overlap |
|---|---:|---:|---:|---:|
| llama31_8b | 0.5943 | 0.1544 | 91.49% | 59.70% |
| qwen3_4b | 0.1514 | 0.1244 | 88.72% | 57.23% |

中心化后交换子明显非零但较小，说明 centered-Key 与 Query 的主方向存在经验对齐，却不能逐方向同时对角化；所以 Key-PCA 奇异向量不等于最优 QK-SVD 奇异向量。沿完整历史均匀采样时，Key 子空间仍较接近 full-history SVD；真实 first-2K basis 的分布失配明显更大。

逐 layer/head 的描述性相关性也没有给出单一可靠代理：删除的行均值能量与 raw QK rank-1 能量在两个模型上的 Spearman 为 0.800/0.866，但 centered commutator、first-2K subspace overlap 或 centered effective rank 都不能跨模型稳定预测生产 fidelity。因此谱量用于构造联合误差账本，不用于替代候选 attention mass 和最终输出验证。

中心化 QK 最优 rank-48 的 p10 为 98.30%/97.74%，但生产 first-2K fidelity 的 p10 只有 39.58%/31.22%。所以 QK 低秩是稳定现象，首段 basis 对困难 head 的准确性不是；最终论证必须继续依赖 attention mass 与 logit/NLL。

### Prefix 长度的纯数值消融

| 模型 | 512 fidelity | 1K | 2K（冻结设置） | 4K | 8K |
|---|---:|---:|---:|---:|---:|
| llama31_8b | 64.30% | 67.02% | 62.39% | 65.04% | 69.21% |
| qwen3_4b | 51.20% | 56.31% | 59.07% | 63.11% | 63.07% |

该表只复用 Q/K trace 改变估计 basis 的首段长度，不改变冻结方法或 LongBench 结果。它用于判断 2K 是否已经进入子空间稳定平台。

Key-PCA48 对 softmax 有效分数的误差严格等于：

$$
\|(S-S_{48})H\|_F^2=\frac{1}{d}\|Q(I-P_{48})(HK)^T\|_F^2.
$$

其中 $H=I-\mathbf1\mathbf1^T/N$。冻结实现使用首 2048 token 的 sampled basis，因此还存在 $Q(P_{48}-\widehat P_{48})(HK)^T/\sqrt d$ 的 prefix-basis 失配项。有效性必须同时来自 Key 尾部奇异值衰减、Query 尾部对齐能量较小，以及首段 basis 对后续历史仍有代表性，而不是“PCA 尾部没有语义”。

RoPE 对不同位置使用不同正交旋转：$C_K=N^{-1}\sum_i R_i\bar k_i\bar k_i^TR_i^T$。因此 pre-RoPE 低秩不自动保证 post-RoPE 首段 basis 可迁移；prefix 长度消融直接检验的正是这项位置条件下的协方差漂移。

令 $E=S-\widehat S$，并令 $\mathcal B_{\gamma,t}$ 是第 $t$ 个 query 在精确 top-$k$ 阈值附近宽度为 $\gamma$ 的边界带。对任意 $\gamma>0$ 有确定性上界：

$$
\frac{1}{MN}\sum_t|S_t^\star\triangle\widehat S_t|\le\frac{2}{MN}\sum_t|\mathcal B_{\gamma,t}|+\frac{2\|E\|_F^2}{MN\gamma^2}.
$$

因此，QK 尾部奇异能量小只控制第二项；还必须同时验证 top-$k$ 边界不承载大量 token/attention mass。完整证明见数学附录。

## 6. Token-logit 稳定性

| 模型 | Token 数 | Top-1 agreement | Margin certificate | KL | JS | 平均 NLL 差 | KL/NLL 界通过率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 3072 | 94.69% | 40.27% | 0.015544 | 0.003487 | +0.010979 | 100.00% |
| Qwen3-4B-Instruct | 3072 | 91.73% | 27.99% | 0.032010 | 0.007700 | +0.006816 | 100.00% |

令 $d=z_{\mathrm{sparse}}-z_{\mathrm{full}}$ 且 $R=\max d-\min d$。当 Full top-1 margin 大于 $R$ 时预测不变；目标 NLL 变化不超过 $R$，并有 $D_{\mathrm{KL}}(p_{\mathrm{full}}\|p_{\mathrm{sparse}})\le R^2/8$。

## 7. 当前请求首段 PCA 基与跨请求固定基

| 模型 | Online macro | Fixed macro | Fixed-Online 95% CI | Fixed/Online | Prediction agreement | 固定基索引构建加速 | 固定基总耗时加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.4582 | 0.4550 | [-0.0086, +0.0025] | 99.29% | 60.00% | 15.382x | 1.144x |
| Qwen3-4B-Instruct | 0.4180 | 0.4196 | [-0.0035, +0.0072] | 100.37% | 41.25% | 15.390x | 1.166x |

这里的固定基由四个与测试样本分离的 LongBench 请求校准，并在全部测试请求中复用。Online 列使用当前请求首个 2048-token chunk 的 post-RoPE 基底；Fixed 列使用四个校准请求的首段 rank-48 二阶矩合并得到的跨请求基底。该消融用于检验 prefix-conditioned basis 是否降低跨请求子空间失配，而不是修改冻结方法。固定基的 Macro 接近 Online，但 prediction agreement 明显较低，而且校准请求仍来自 LongBench；在独立语料和未见模型上验证之前，它只能作为减少短上下文索引构建开销的候选优化，不能写成无需校准的通用替代。

## 8. 结论与限制

1. 结论仅支持自然输入分布上的条件稳定性，不是任意 query 的无条件证明。
2. 7.5K LongBench 的主要作用是质量与跨模型验证；短文本速度仍是弱项。
3. 当前 final-direct LongBench 保留完整 GPU K/V，不能把 attention 比例写成 GPU KV 比例。
4. PCA/SVD top-k 本身不是新颖点；论文贡献必须落在当前请求首段条件化的低比特 direct retrieval、误差账本、无 Full 回退和真实长上下文系统实现的组合上。
