# 纯数值 KV 检索优化：QK-Metric 低秩索引与 128K 结果

更新时间：2026-07-21

## 1. 本轮目标

当前系统已经验证：对每个 query head，只让最终 attention 使用原始 KV 中 exact top-2% 的 token，在长上下文上可以保持甚至改善 PPL。真正的问题变成：如何用更小、更快的索引找到这 2%，而不是继续训练 router 或反复手调任务预算。

本轮只研究不依赖任务标签、答案、oracle 和学习模块的数值方法，并同时统计：

- top-2% 位置召回和 attention mass 召回；
- 128K 真实 PPL；
- 包含检索、两次 top-k、exact rerank 和最终 V attention 的子系统速度；
- 包含一次性索引构建/重建的 1024-token 整模型在线速度。

## 2. PDF 带来的机制启发

`why_long_context_hurts_needle_retrieval_softmax_rope_20260720.pdf` 的实验表明，长上下文退化同时来自：

1. 上下文增长后 softmax 分母增大，相关 token 的 attention mass 被大量干扰 token 稀释；
2. RoPE 相对位置相位会继续改变远距离 QK logit。

这解释了为什么 exact top-2% 有时比 Full attention 的 PPL 更低：稀疏 attention 不只是压缩，也在删除 softmax 竞争中的低价值干扰项。但它不直接解决“怎样快速找到 top-2%”。

## 3. 先得到的工程前沿

普通 post-RoPE PCA64 + logscale16 INT4 索引原来扫描 8% 候选，再用原始 K exact rerank 到 2%。在独立 QKV trace 上缩小候选池后：

| 近似候选池 | top-2% attention mass 召回均值 | 最差 head | 128K 子系统速度 |
|---:|---:|---:|---:|
| 5% | 99.9290% | 98.6306% | 2.88x |
| 6% | 99.9634% | 99.1897% | **2.93x** |
| 7% | 99.9797% | 99.3865% | 2.93x |
| 8% | 99.9879% | 99.5188% | 2.73x |

普通 PCA64 candidate-6% 的四主题 128K PPL 为 `10.3658`，平均在线速度约 `2.53x`。因此 6% 是比旧 8% 更好的实用点，但它仍然让低维空间最小化 K 重建误差，没有直接优化 QK 排名。

## 4. QK-Metric 成对低秩空间

### 4.1 为什么 PCA 目标不对齐

PCA 求一个对称投影，使 K 的重建误差尽量小：

```text
minimize  E[ ||k - A k||^2 ]
```

KV 检索真正关心的是 attention score `q^T k`。某个 K 方向方差很大，但 query 几乎不用它，PCA 仍可能把有限维度分给它；反过来，一个 K 方差不大但对当前 query 很重要的方向可能被丢弃。

### 4.2 新目标

令 query/key 二阶矩分别为 `Cq` 和 `Ck`，寻找 rank-r 矩阵 `A`，直接最小化 QK score 的均方误差：

```text
minimize_A  E[ (q^T k - q^T A k)^2 ]
subject to  rank(A) <= r

= minimize_A  || Cq^(1/2) (I - A) Ck^(1/2) ||_F^2
```

对矩阵 `Cq^(1/2) Ck^(1/2)` 做 SVD：

```text
Cq^(1/2) Ck^(1/2) = U Sigma V^T
```

取前 r 个奇异方向并拆成 query/key 两个不对称因子：

```text
L = Cq^(-1/2) U_r Sigma_r^(1/2)
R = Ck^(-1/2) V_r Sigma_r^(1/2)

q^T k  approximately equals  (q^T L) (k^T R)^T
```

这不是学习模块。`Ck` 从 post-RoPE K 每 32 个 token 采样估计；每个 KV head 用最开始 4 个在线 query 估计 `Cq`，并用 0.5 isotropic shrinkage 稳定低样本协方差。

### 4.3 真实运行流程

```text
前 3 个 decode query：普通 PCA64 INT4 检索
第 4 个 query：得到 Cq，闭式计算 QK-Metric 的 L/R
             用 R 一次性重建历史 K 的 INT4 索引
后续 query：q @ L -> INT8 query
           扫描 K @ R 的 logscale16 INT4 索引
           取近似 top-6% 候选
           原始 K exact rerank 到 top-2%
           原始 V 做 split-parallel sparse attention
```

校准完成后会立即释放 `Cq/Ck`，常驻状态只增加一个很小的 query 投影矩阵。

## 5. 独立召回结果

### 5.1 Llama 跨主题、跨层 held-out query

Llama-3.1-8B，sports + medicine，层 `0/8/16/24/31`；前 4 个 query 校准，query 8-15 测试，共 2560 个 query-head case。普通 PCA 不依赖 query 校准，因此使用同一组测试 query：

| 方法，candidate-6% | top-2% 位置召回 | attention mass 召回 | 最差 mass |
|---|---:|---:|---:|
| 普通 PCA64 INT4 | 96.750% | 99.9050% | 94.4907% |
| **QK-Metric64 INT4** | **99.453%** | **99.9870%** | **94.5988%** |

QK-Metric 的主要收益是把普通 case 的近似排序误差大幅压低。最差 head 只比普通 PCA 略好，仍未完全解决，因此当前不把候选池直接降到 4%。同条件下，rank-48/candidate-6% 的平均 mass 为 `99.9590%`、最差为 `92.4261%`；rank-64/candidate-4% 的平均 mass 为 `99.9580%`、最差为 `92.6924%`。二者都不适合作为通用默认值。

### 5.2 Qwen 128K layer-16

前 4 个 query 校准，query 8-15 独立测试：

| 数值表示，candidate-6% | top-2% 位置召回 | attention mass 召回 | 最差 mass |
|---|---:|---:|---:|
| QK-Metric32 INT4 | 98.600% | 99.9302% | 96.9865% |
| QK-Metric48 INT4 | 99.685% | 99.9943% | 99.8875% |
| **QK-Metric64 INT4** | **99.903%** | **99.9983%** | **99.9581%** |
| QK-Metric64，前 32 维 INT4 + 后 32 维 INT2 | 99.827% | 99.9966% | 99.9027% |

rank-48 和谱尾 INT2 都值得继续做真实 PPL，但 rank-32 的最差 case 已明显退化。

在每个 16 维带内加入可折叠的 Hadamard 旋转只有约 `0.0003` 个百分点的 mass 收益，且没有稳定改善最差 case；当前不加入主方法。

## 6. 128K 整模型质量与速度

设置：Qwen3-4B-Instruct，2×RTX 3090 分层放置，128K history，每主题 1024 个逐 token 预测。在线时间不含 prefill，但包含索引建立、4-query 校准、36 层 QK-Metric 索引重建、检索和模型其余计算。

| 主题 | Full PPL | 普通 PCA64 c6 PPL | QK-Metric64 c6 PPL | QK-Metric 在线时间 | 对 Full 加速 |
|---|---:|---:|---:|---:|---:|
| Medicine | 9.9795 | 9.9400 | **9.8931** | 227.685 s | 2.512x |
| Politics | 12.3210 | 12.1666 | **12.0802** | 228.027 s | 2.508x |
| Computer | 7.4658 | 7.4186 | **7.3742** | 227.957 s | 2.510x |
| Space | 12.8824 | 12.8687 | **12.8106** | 228.047 s | 2.510x |
| **几何平均/平均时间** | **10.4281** | **10.3658** | **10.3080** | **227.929 s** | **2.510x** |

结论：

- 相对 Full，QK-Metric 的 PPL 质量保持率为 `101.17%`；
- 相对普通 PCA candidate-6%，QK-Metric 的几何 PPL 再改善 `0.56%`；
- 四个主题全部改善，不是由单一主题拉动；
- 普通 PCA candidate-6% 平均约 226.3 秒，QK-Metric 多约 1.6 秒，基本等于 36 层的一次性重建成本。

### 6.1 Attention 子系统

相同 Qwen 128K layer-16 trace、RTX 3090：

| 方法 | 每层每 token | 相对 Full |
|---|---:|---:|
| Full SDPA | 2.3973 ms | 1.000x |
| 普通 PCA64 candidate-6% | 0.8374 ms | 2.863x |
| **QK-Metric64 candidate-6% 稳态** | **0.8476 ms** | **2.828x** |
| QK-Metric48 candidate-6% 稳态 | 0.8373 ms | 2.863x |
| QK-Metric64 candidate-4% 稳态 | 0.8325 ms | 2.879x |

表中是 3 张 RTX 3090、每张 20 个测量 cycle 的中位数。QK-Metric 不改变 kernel 形状，但改变候选 token 的物理地址；QK64/6% 在这组 trace 上因 gather 行为比普通 PCA 慢约 1.2%，整模型差距更小。无 warmup 的 16-query harness 把第 4 步重建计入后，额外成本约为：

```text
=(3.6312 - 0.8476) ms/query * 16 queries
= 44.6 ms / layer，一次性
```

36 层约为 1.6 秒，与完整端到端实测差值一致。

### 6.2 Rank-48 高效点

同一四主题 128K 实验把 QK-Metric 从 64 维改为 48 维，候选池和最终 attention 仍为 6%/2%：

| 主题 | Rank-64 PPL | Rank-48 PPL | Rank-48 在线时间 | Rank-48 对 Full 加速 |
|---|---:|---:|---:|---:|
| Medicine | 9.8931 | **9.8920** | 223.439 s | 2.559x |
| Politics | **12.0802** | 12.0928 | 223.527 s | 2.558x |
| Computer | **7.3742** | 7.3788 | 223.638 s | 2.559x |
| Space | **12.8106** | 12.8204 | 223.763 s | 2.558x |
| **几何平均/平均时间** | **10.3080** | **10.3139** | **223.592 s** | **2.559x** |

Rank-48 相对 Rank-64 的 PPL 只差 `0.058%`，仍比 Full 好 `1.11%`；平均在线时间再下降 `4.34s`，索引常驻比例约从 7.2% 降到 5.6%。但 Llama 跨层最差 mass 从 Rank-64 的 94.60% 降到 92.43%，因此它是更好的 Qwen 高效点，不是当前最稳妥的跨模型默认值。

Rank-64/candidate-4% 也完成了相同实验：几何 PPL `10.3109`、平均在线时间 `224.871s`、加速 `2.544x`。它没有 Rank-48/6% 的小索引和速度，跨层最差 mass 又只有 92.69%，因此不保留为主 operating point。

## 7. 三种比例不能混淆

当前 QK-Metric64 主方法同时有三个不同概念：

| 比例 | 含义 |
|---:|---|
| 约 7.1% | PCA64 logscale16 INT4 K 索引相对原始 FP16 K+V 的常驻字节比例 |
| 6% | 每个 query head 从近似分数中送入 exact QK rerank 的候选 token 比例 |
| 2% | exact rerank 后真正参与最终 attention 的原始 K/V token 比例 |

不能把 2% attention link ratio 写成总 GPU KV 占用，也不能把 6% 候选池写成最终 attention 比例。

128K harness 计入 capacity 和投影矩阵后的实际 state ratio 为：QK-Metric64 `7.197%`，QK-Metric48 `5.596%`。

## 8. 本轮否定的方向

| 方向 | 结论 |
|---|---|
| sampled current-score quantile | 第一阶段 scan 更快，但不规则候选数和 ragged exact rerank 使完整子系统从约 2.72x 降到 2.28x |
| 局部时间 tile quota | attention mass 只有约 79%-90%，破坏全局极值竞争 |
| block upper bound pruning | 需要扫描约 84%-97.5% block，界太松 |
| previous-query threshold | query 间标准化阈值不稳定，最差召回接近 0 |
| Hadamard / 8-seed 校准旋转 | 可折叠、零稳态开销，但收益过小，不足以增加方法复杂度 |
| rank-32 | 平均尚可，但 128K 最差 mass 只有约 96.99%，暂不进入真实 PPL 主线 |
| score-extreme 加权 Ck | Llama 最差值偶有改善，但平均召回下降，Qwen 也没有稳定收益 |
| coverage 谱门控 | Llama coverage-0.98 用 30% rank-64 层恢复最差值，但同阈值在 Qwen 过度保守地全选 rank-64，尚不跨模型通用 |

这些失败说明：当前瓶颈不是“再聪明一点地猜阈值”，而是低维空间本身是否与 QK score 对齐。QK-Metric 在这一点上给出了闭式、通用且可验证的改进。

## 9. 当前推荐与下一步

当前论文主方法建议使用：

```text
QK-Metric64 + logscale16 INT4 K index
-> fixed candidate-6%
-> original-K exact rerank
-> fixed final top-2%
-> hardware-aware split sparse attention
```

它比普通 PCA 多一个明确的数值创新，同时已经在真实 128K PPL 上得到四主题一致改善，并保持约 2.51x 整模型加速。

同时报告一个不隐藏风险的效率 operating point：`QK-Metric48 / candidate-6% / final-2%`，其 128K Qwen 实测为约 5.6% 索引、2.559x 在线加速和 101.11% Full PPL 质量；跨模型主表仍以 Rank-64 为默认。

下一步优先级：

1. 冻结 QK-Metric64/6% 为跨模型质量默认点，QK-Metric48/6% 为效率点，不再继续微调 rank 和候选比例；
2. 在完整 Llama/Qwen 多层 trace、更多随机文本和不同长度上验证两个 operating point；
3. 进入公开 LongBench、RULER、Long ICL 和生成任务主表，补 Full、普通 PCA 与已发表 KV baseline；
4. 单独分析 Llama layer-8 的最差 head，只有找到跨模型闭式风险量后才考虑谱门控，不训练 router。

## 10. 复现入口

- 核心实现：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_head_top2_targeted_ppl_20260714.py`
- PPL 入口：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_adaptive_mass_budget_ppl_20260715.py`
- QK-Metric 离线验证：`ymluo/projects/qwen3_top2_token_mechanism/src/analyze_qk_metric_lowrank.py`
- rank/旋转/精度前沿：`ymluo/projects/qwen3_top2_token_mechanism/src/analyze_qk_metric_rotation_precision.py`
- 128K PPL：`ymluo/projects/qwen3_top2_token_mechanism/scripts/run_qkmetric_128k_20260721.sh`
- 后续运行矩阵：`ymluo/projects/qwen3_top2_token_mechanism/scripts/launch_qkmetric_runtime_frontier_20260721.sh`
