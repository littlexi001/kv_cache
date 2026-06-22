# 当前验证结果

## 0. 当前核心结论

对某一层某个 attention head，最终 score 为：

$$
s_{ij}
=
\frac{
\operatorname{Norm}(W_Qx_i)^{\top}
R_{i-j}
\operatorname{Norm}(W_Kx_j)
}{\sqrt d}
$$

现有结果支持如下计算链条：

1. layer input 中已经存在与 top-2% key selection 相关的内容 feature；
2. `W_Q/W_K` 对这些 feature 做 head-specific 的选择和放大；
3. RoPE 根据相对位置进一步抑制无关 token，尤其是无关的远距离 token；
4. K 的 common mean 占据大量能量并抬高 raw cosine，但该均值在 token-wise softmax 中抵消，不是有效 retrieval feature；
5. 去除 common mean 后，top-2% token 的 feature 相似性仍然显著存在，而且区分度更强；
6. 有效 K feature 主要集中在前部奇异方向，但不是只由少数最大奇异值方向完成，而是“前部核心方向 + 数百个长尾修正方向”。

由于 input、Q/K transformation 和 RoPE 在公式中是乘性交互，现有实验不能把它们写成三个天然可加且对所有 layer/head 通用的百分比。当前可以报告的是基于明确对照的条件贡献比例。

## 1. 实验设置

本轮分析使用：

1. 本地预训练 `Qwen3-0.6B`；
2. 一条长度超过 5000 token、前部埋藏事实并在后部提问的 synthetic long-QA 序列；
3. query token 位置 `5000-5099`；
4. 全部 28 层、每层全部 16 个 query head；
5. 每个 query/head 的 strict-history top-2% key 作为正样本；
6. 随机历史 token 和距离匹配 token 作为两类对照；
7. Layer `0/7/13/20/27` 对 K 做完整 1024-rank 精确 SVD，每层分析全部 head；
8. K 方向归因直接重构模型真实的 K RMSNorm、RoPE 和 QK score。

完整 K 方向贡献对真实 QK score 的平均重构误差约为 `0.0016`，相对于 score 量级足够小。

结果文件：`outputs/qk-rope-full-k-direction-qwen3-0.6b/qk_feature_svd_summary.json`。

## 2. Layer Input 与 K Common Mean

### 2.1 Common mean 严重污染 raw K cosine

最终 RoPE 后 K-space 中，common mean 能量占比：

| 指标 | 数值 |
|---|---:|
| 平均值 | 65.5% |
| 中位数 | 68.6% |
| 90% 分位 | 91.9% |

common-mean 能量与随机 token raw K cosine 的相关系数为 `0.985`，与 top-2% raw K cosine 的相关系数为 `0.925`。因此 raw K cosine 普遍很高主要来自 shared center，不能直接解释 retrieval feature。

### 2.2 去中心后，top-2% feature 更清楚

| Pair 类型 | Raw K cosine | Centered K cosine |
|---|---:|---:|
| top-2% | 0.729 | 0.396 |
| 距离匹配 | 0.691 | 0.308 |
| 随机历史 token | 0.615 | -0.012 |
| top - random | 0.114 | 0.408 |
| top - distance-matched | 0.038 | 0.088 |

去中心后，随机 token 的 cosine 回到约零，而 top-2% 仍保持明显正相似性。全部 layer/head 都满足 centered top-2% K cosine 高于随机 token，`92.2%` 的 layer/head 高于距离匹配对照。

这说明 common mean 没有制造 top-2% feature；它反而遮蔽了真正具有区分度的 residual K geometry。

## 3. Input Feature 与 Q/K Transformation

以随机历史 token 为对照：

| 表征空间 | Centered top-random gap |
|---|---:|
| Layer-input hidden | 0.236 |
| K space | 0.408 |

仅在“centered representation cosine 区分度”这个指标内，可以写成：

1. layer input 已有 feature 约占最终 K-space gap 的 `57.8%`；
2. K transformation 额外放大约 `42.2%`。

以距离匹配 token 为对照：

| 表征空间 | Centered top-distance gap |
|---|---:|
| Layer-input hidden | 0.0668 |
| K space | 0.0884 |

对应的条件比例为：

1. layer input 已有 feature：约 `75.6%`；
2. K transformation 增量：约 `24.4%`。

这些比例只描述 hidden-to-K representation geometry。它们没有完整包含 query-side `W_Q`，因此不能直接作为最终 attention score 的 input/QK 百分比。

## 4. 完整 K 奇异方向归因

按奇异值排名将完整 1024 个 K 方向等分为四段：

| K 奇异方向 | 绝对 top-random 区分贡献 | 净正向 score margin |
|---|---:|---:|
| 0-255 | 68.6% | 86.5% |
| 256-511 | 15.2% | 9.8% |
| 512-767 | 10.4% | 3.2% |
| 768-1023 | 5.8% | 0.6% |

因此，有效 K feature 的净正向信息主要来自前 256 个奇异方向。此前仅比较 top/random 各自归一化后的谱带质量，两组都大量使用高能方向，因而掩盖了 contribution 的正负差异；完整 signed direction attribution 修正了这一判断。

但奇异值大小并不完全决定 feature 重要性：

1. 奇异值与方向区分贡献的平均相关系数约为 `0.445`；
2. 平均约 `43` 个方向覆盖 50% 的绝对区分贡献；
3. 平均约 `404` 个方向才能覆盖 90%；
4. 不同层覆盖 50% 所需方向分别约为：Layer 0=`3`、Layer 7=`50`、Layer 13=`54`、Layer 20=`83`、Layer 27=`25`。

因此更准确的结构是：少量前部奇异方向形成核心 feature，大量中低能方向提供 layer/head-specific 的长尾修正。

## 5. RoPE Ablation

### 5.1 RoPE 显著改变 top-2% membership

移除 RoPE 后，新的 top-2% 集合与完整模型原 top-2% 集合平均 overlap 仅为 `42.2%`。约 58% 的 top-key membership 依赖 RoPE。

### 5.2 RoPE 对 score margin 的条件贡献

固定完整模型选出的原 top-2% key，不重新选择正样本。

| 对照 | 完整 RoPE margin | 无 RoPE margin | RoPE 条件增量 |
|---|---:|---:|---:|
| 随机历史 token | 8.40 | 3.94 | 53.1% |
| 距离匹配 token | 3.21 | 2.44 | 24.1% |

两个比例回答不同问题：

1. `53.1%` 表示 RoPE 在全历史 KV candidate 中的实际筛选作用；
2. `24.1%` 表示已经控制相对距离后，RoPE 仍提供的额外区分。

### 5.3 RoPE 的主要作用是抑制无关 token

移除 RoPE 后：

| Pair 类型 | 完整 RoPE score | 无 RoPE score |
|---|---:|---:|
| top-2% | 6.36 | 8.33 |
| 随机历史 token | -2.05 | 4.39 |

RoPE 并不是简单提高相关 token 的 score。它会降低两组 score，但对随机 token 降得更多：

| Pair 类型 | RoPE score change |
|---|---:|
| top-2% | -1.98 |
| 随机历史 token | -6.44 |

这种抑制随距离增强：

| 距离 | top-2% change | random change |
|---|---:|---:|
| 1-16 | -0.73 | -1.25 |
| 65-256 | -2.26 | -3.82 |
| >1024 | -2.27 | -6.78 |

因此 RoPE 更像相对位置相关的过滤器：它强烈抑制无关远距离 token，同时允许少量内容相关的远距离 token 保持相对高分。

## 6. 三部分贡献应如何表述

当前不能把不同指标上的比例拼成一个 `Input + QK + RoPE = 100%` 的数字。原因是：

1. input 与 `W_Q/W_K` 是乘法关系，没有 input 就不存在 transformation 输出；
2. RoPE 与 Q/K 也是乘法关系，其效果依赖当前 query/key feature 方向；
3. 不同 layer/head 的条件贡献差异很大；
4. hidden/K cosine 与最终 QK score 不是同一个测量尺度。

当前可报告的系统性结论是：

| 阶段 | 当前结论 |
|---|---|
| Layer input | 已经具有可测的 top-key feature geometry |
| Q/K transformation | 对已有 feature 做 head-specific 选择和放大；K 前部谱方向贡献主要净正向 margin |
| RoPE | 负责显著的相对位置过滤；相对随机候选解释约一半 margin，距离匹配后解释约四分之一 |

若需要在同一 attention-score metric 上给出三项统一贡献，下一步应完成 `input relationship × learned QK × RoPE` 的八组 factorial ablation，并用 Shapley value 分配主效应与交互项。

## 7. Claim Boundary

本轮结果允许声称：

1. top-2% key 不是由 K common mean 产生的虚假集合；
2. feature 在 layer input 中已有体现，并被 Q/K transformation 放大；
3. 有效 K feature 偏向前部奇异方向，但仍具有明显长尾；
4. RoPE 对最终 retrieval set 和 score margin 都有重要贡献；
5. top-key selection 是内容 feature、learned Q/K geometry 与相对位置共同作用的结果。

本轮结果不允许声称：

1. 三部分存在对所有层通用的唯一加法百分比；
2. 结果已经在多模型、多真实语料上复现；
3. top-2% feature 已经能够由 pre-attention router 准确预测；
4. feature bucket 已经转化为实际 KV memory 或 latency 收益。

当前最小剩余不确定性是：在统一 score metric 和统一 baseline 下，input relationship、learned Q/K orientation、RoPE 及其交互项分别解释多少 top-key margin。

## 8. 架构实现 Micro-test

现有本地 micro-test 已验证：

1. detached exclusive causal mean 不会向历史 router states 传播梯度；
2. 修改未来 token 不会改变更早位置的 logits；
3. `layer_input/q/k/v` router input 均支持 forward/backward；
4. NTP gradient 能到达 router 参数；
5. checkpoint、runtime config、JSONL metrics 和 summary 可以端到端生成；
6. 完整 Qwen3-0.6B 配置低于 2B 参数限制。

当前 equal-budget 参数检查：

| Architecture | Total parameters |
|---|---:|
| ordinary top-1 MoE, `1024 -> 3072 -> 1024` | 1.388888B |
| shared full-output head MoE, `64 -> 512 -> 1024` | 1.389003B |

这些检查只证明代码路径和预算匹配，不证明 bucket specialization、DCLM quality 或实际 KV-memory/latency 收益。
