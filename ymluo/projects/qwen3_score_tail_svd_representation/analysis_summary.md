# Qwen3 Score-Tail SVD 表示分析 — 结果总结

## 项目概述

该项目对 **Qwen3-0.6B** 模型的全 28 层、全 16 个 attention head 进行了 SVD（奇异值分解）表示分析。输入使用 5000 个 prefill token + 1024 个 eval token。核心目标是**比较按 QK score 排名划分的三组 token 在 K/V/weighted-V 三个表示空间中的几何分布差异**。

### 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B |
| Prefill Tokens | 5000 |
| Eval Tokens | 1024 |
| Chunk Size | 128 |
| 层数 | 0–27（全部 28 层） |
| 头数 | 0–15（全部 16 头） |
| 表示类型 | key, value, weighted_value |
| SVD 成分数 | 8 |
| Query Stride | 8 |
| 每组最大向量数 | 4096 |

---

## 三组 Token 定义

| 分组 | 含义 | 平均 QK Score | 平均 Attention Weight |
|------|------|--------------|---------------------|
| **score_top_1pct** | QK score 最高的 1% token | **+7.18** | **1.57×10⁻²** (主导注意力) |
| **score_top_90pct** | QK score 最高的 90% token | −0.47 | 2.02×10⁻⁴ |
| **score_tail_10pct** | QK score 最低的 10% token | **−6.18** | **1.14×10⁻⁷** (几乎不参与注意力) |

> tail_10pct 的 attention weight 比 top_1pct 低 **5 个数量级**，说明这些 token 在最终的加权输出中几乎不存在。

---

## 关键发现

### 1. SVD 能量集中度（singular_value_energy.csv）

所有组共享同一个 SVD 基底，PC1 的能量集中度反映了该表示空间的**低秩程度**：

| 表示空间 | 平均 PC1 能量占比 | PC1/PC8 能量比 | 解读 |
|---------|-----------------|---------------|------|
| **key** | 21.3% | 3.17× | 整体高秩，早期层 PC1 占 90%，后期层仅 ~40% |
| **value** | 10.0% | 1.86× | **最均匀分布**，信息分散在多个方向 |
| **weighted_value** | 27.1% | 3.19× | **最集中的能量**，第一主成分主导 |

> **每层间差异巨大**：以 key 为例，Layer 0 的 PC1 占 **91.5%**，而 Layer 2 仅占 **40.2%**。早期层的表示远更集中。

#### 各 PC 成分平均能量占比

| PC 成分 | key | value | weighted_value |
|---------|-----|-------|---------------|
| PC1 | 21.3% | 10.0% | 27.1% |
| PC2 | 9.2% | 5.0% | 12.7% |
| PC3 | 6.4% | 3.8% | 8.1% |
| PC4 | 5.1% | 3.2% | 6.0% |
| PC5 | 4.1% | 2.8% | 4.7% |
| PC6 | 3.5% | 2.5% | 3.9% |
| PC7 | 3.1% | 2.3% | 3.3% |
| PC8 | 2.7% | 2.1% | 2.8% |

---

### 2. 分组在各 PC 方向上的投影能量（svd_projection_by_group.csv）

关键指标是 `pc1_energy_ratio`——分组向量集投影到 PC1 方向上占该组总能量的比例：

#### key 空间

| 分组 | PC1 能量占比 | 投影模长 |
|------|------------|---------|
| score_top_1pct | **33.5%** | 19.13 |
| score_tail_10pct | **32.0%** | 19.33 |
| score_top_90pct | 23.5% | 16.84 |

#### value 空间

| 分组 | PC1 能量占比 | 投影模长 |
|------|------------|---------|
| score_top_1pct | **22.3%** | 13.23 |
| score_tail_10pct | 21.1% | 12.07 |
| score_top_90pct | 18.3% | 10.94 |

#### weighted_value 空间（最关键的差异）

| 分组 | PC1 能量占比 | 投影模长 |
|------|------------|---------|
| score_top_1pct | **33.0%** | **0.124** |
| score_tail_10pct | **51.9%** | **1.9×10⁻⁵** |
| score_top_90pct | 51.6% | 0.020 |

> **weighted_value 空间的核心结论**：tail_10pct 的投影模长比 top_1pct 小 **~6500 倍**，但它在这种近乎"零向量"的状态下 PC1 占比仍然高达 52%（可能是噪声地板的表象）。top_90pct 由于包含了绝大部分 token，PC1 占比也高（51.5%），但 top_1pct 的 PC1 占比只有 33%，说明高 score token 的加权值分布在多个方向上。

---

### 3. 质心余弦相似度（centroid_similarity_by_group.csv）

| 比较对 | key | value | weighted_value |
|--------|-----|-------|---------------|
| top_1pct ↔ top_90pct | **0.942** | **0.787** | **0.569** |
| tail_10pct ↔ top_90pct | 0.880 | 0.785 | 0.188 |
| tail_10pct ↔ top_1pct | 0.798 | 0.534 | 0.366 |

> **key 空间**：三组几乎指向同一方向（余弦相似度 0.80–0.94），说明 key 空间中所有 token 共享一个主导方向。
>
> **value 空间**：tail_10pct 和 top_1pct 的相似度仅 **0.534**，说明 low-score token 和 high-score token 的值向量的主要方向差异较大。
>
> **weighted_value 空间**：tail_10pct 和 top_90pct 几乎正交（**0.188**），这印证了 tail_10pct 在 attention 加权后基本没有有效信号。

---

### 4. Score 分布特征（score_distribution_by_group.csv）

- **top_1pct** 的 score 平均值范围跨 head 差异很大（从 −1.49 到 +29.74），不同 head 的高分 token 的"绝对 quality"差异显著。
- **tail_10pct** 的 score 始终为负且集中在 −18.96 到 +0.09 之间。
- **top_90pct** 的 score 分布覆盖范围最广（从 −20.38 到 +20.10），因其包含了绝大部分 token。

---

## 总体结论

1. **加权值 (weighted_value) 是最具区分力的表示空间**：top_1pct 和 tail_10pct 的投影模长差 4 个数量级，tail_10pct 几乎不贡献任何有效信号。从 KV cache 压缩的角度看，这些 token 可以安全地丢弃。

2. **Key 空间的信息高度方向性**：PC1 在全部分组中都占主导，且三组质心方向高度一致（余弦 > 0.80），不同层之间差异较大。

3. **Value 空间的能量分布最均匀**：PC1 仅占 10%，信息分散在多个方向。low-score token 与 high-score token 在 value 空间中方向差异最大（余弦相似度仅 0.534）。

4. **早期层的 SVD 能量集中度远高于晚期层**：Layer 0 的 key PC1 占 91%，而 Layer 2+ 仅 ~40%，说明深层需要注意更多、更分散的方向。

5. **Tail 10% 的 token 在加权后可完全忽略**：它们的 weighted_value 投影模长仅为 top_1pct 的 ~1/6500，从表示角度看，KV cache 压缩策略应优先保留 top-scoring token。

6. **三种表示空间各有特点**，对 KV cache 压缩的启示：
   - **key**：所有 token 方向一致，可以按单一主方向进行压缩。
   - **value**：信息均匀分散，压缩时需要保留更多方向。
   - **weighted_value**：tail token 的贡献极小，可以激进地通过 score 阈值过滤。

---

## 输出文件清单

| 文件 | 行数 | 描述 |
|------|------|------|
| `svd_projection_by_group.csv` | 4032 | 每组在各 PC 上的投影统计 |
| `centroid_similarity_by_group.csv` | 4032 | 组间质心余弦相似度与 L2 距离 |
| `score_distribution_by_group.csv` | 1344 | 各组的 QK score 和 attention weight 分布 |
| `score_top_90pct_distribution.csv` | 448 | top_90pct 组的专门分布统计 |
| `singular_value_energy.csv` | 10752 | 各层各头的全局 SVD 奇异值能量 |
| `summary.json` | — | 运行配置与输出路径汇总 |
