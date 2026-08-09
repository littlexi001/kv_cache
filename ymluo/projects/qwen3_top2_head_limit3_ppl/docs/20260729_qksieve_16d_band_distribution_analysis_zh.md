# QKSieve 八个 16 维频带的数值分布分析

## 1. 要回答的问题

QKSieve 将每个 head 的 128 维 Q/K 表征划分为 8 个连续的 16 维频带：

| 频带 | 维度 |
|---|---|
| band 1 | 0--15 |
| band 2 | 16--31 |
| band 3 | 32--47 |
| band 4 | 48--63 |
| band 5 | 64--79 |
| band 6 | 80--95 |
| band 7 | 96--111 |
| band 8 | 112--127 |

这里需要区分三个问题：

1. 前部 band 的投影后 K 数值是否更大；
2. 前部 band 的 held-out Q 数值是否更大；
3. 前部 band 对真正的 QK score 是否贡献更大。

第三项最重要。只看 K 的绝对值可能误判，因为一个可逆双线性变换可以同时缩小
K、放大 Q，而不改变二者乘积。

## 2. 理论预测

记正则化后的 Query 二阶矩为 \(C_q\)，Key 二阶矩为 \(C_k\)。QK-balanced
变换先分解：

\[
C_q^{1/2} C_k^{1/2}=U\Sigma V^\mathsf{T},
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_{128}),
\]

其中 \(\sigma_1\ge\cdots\ge\sigma_{128}\)。然后使用：

\[
A=C_q^{-1/2}U\Sigma^{1/2},
\qquad
B=C_k^{-1/2}V\Sigma^{1/2}.
\]

投影后的 Query 和 Key 分别为 \(q'=qA\)、\(k'=kB\)。在校准二阶矩下：

\[
A^\mathsf{T}C_qA=\Sigma,
\qquad
B^\mathsf{T}C_kB=\Sigma.
\]

因此第 \(j\) 个投影坐标的 Q 方差和 K 方差都等于 \(\sigma_j\)，该坐标的
QK score 二阶贡献与 \(\sigma_j^2\) 成正比。由于奇异值已经降序排列：

- 前部 16 维 band 的平均 RMS 理论上更大；
- band score 能量在校准分布上单调不增；
- 这不是原始维度编号的性质，而是 QK-balanced 变换得到的共同谱排序。

这项单调性对校准二阶矩严格成立，但对 held-out Query 的逐样本绝对值并非定理。
因此还需要真实轨迹验证分布漂移、极端值和 top-2% token 上的贡献。

## 3. 已有跨模型实证

已有结果覆盖 Qwen3-4B、Qwen2.5-7B 和 Llama-3.1-8B，包含 sports 和
medicine 两个 32K 主题，共 200 个 layer/KV-head 条件。

### 3.1 与当前 QK-balanced 校准一致的累计结果

| 指标 | 平均值 |
|---|---:|
| band 1 的 QK score 能量 | 91.22% |
| bands 1--3 的 QK score 能量 | 96.57% |
| bands 4--8 的 QK score 能量 | 3.43% |

分模型和主题的结果：

| 模型与主题 | band 1 | bands 1--3 | bands 4--8 |
|---|---:|---:|---:|
| Llama-3.1-8B medicine | 92.97% | 97.31% | 2.69% |
| Llama-3.1-8B sports | 93.06% | 97.36% | 2.64% |
| Qwen2.5-7B medicine | 93.16% | 97.40% | 2.60% |
| Qwen2.5-7B sports | 93.52% | 97.54% | 2.46% |
| Qwen3-4B medicine | 88.33% | 95.34% | 4.66% |
| Qwen3-4B sports | 88.38% | 95.40% | 4.60% |

结论在三个模型和两个主题间一致，但 Qwen3-4B 的谱尾明显比另外两个模型更重。

### 3.2 去除 softmax 无效均值模态后的结果

另一份完整中心化 QK 谱分析覆盖 160 个 layer/KV-head 条件。按每 16 维的
累计谱能量差分后：

| 频带 | 中心化 QK score 能量 |
|---|---:|
| band 1 | 94.149% |
| band 2 | 3.399% |
| band 3 | 1.257% |
| band 4 | 0.605% |
| bands 5--8 合计 | 0.590% |

中心化后仍然存在非常强的头重脚轻分布，所以该现象不只是由逐 Query 常数
score 模态造成。

### 3.3 自动位宽分配的间接验证

在相同的 200 个 QK-balanced layer/KV-head 条件上，固定总物理预算的 qMSE
分配得到：

| 频带 | 平均 bit | 0-bit 比例 | 至少 4-bit 比例 |
|---|---:|---:|---:|
| band 1 | 4.640 | 0.0% | 100.0% |
| band 2 | 3.730 | 0.0% | 91.0% |
| band 3 | 2.190 | 7.0% | 34.5% |
| band 4 | 0.725 | 41.5% | 0.0% |
| band 5 | 0.000 | 100.0% | 0.0% |
| band 6 | 0.000 | 100.0% | 0.0% |
| band 7 | 0.000 | 100.0% | 0.0% |
| band 8 | 0.000 | 100.0% | 0.0% |

这不是独立的质量证明，因为 allocator 本身使用频带 qMSE；但它与共同谱能量的
快速衰减方向一致。

## 4. 当前可以下的结论

可以确认：

1. 在 QK-balanced 坐标中，前部 band 的典型幅值和 score 二阶贡献应当更大；
2. 跨模型真实轨迹确认 score 能量高度集中在前 1--3 个 band；
3. 末尾 band 的平均贡献很小，但不能表述为“所有末尾数值都接近零”；
4. 平均能量小仍可能伴随少量大离群值，后者可能影响 top-k 边界 token。

因此，当前结果支持 mixed-bit 分配，不直接支持无条件删除全部谱尾。

## 5. 逐频带分布实验

新增分析脚本：

`src/analyze_qksieve_band_distributions_20260729.py`

它在同一批三模型 32K 轨迹上分别统计：

- K 的 mean-absolute、RMS、median、P90、P99、max 和能量占比；
- held-out Q 的相同统计；
- 每个 band 的 QK score mean-absolute、RMS、方差占比；
- exact total-score top-2% token 上每个 band 的绝对贡献；
- 原始 128 维直接切块的对照结果。

原始切块对照用于确认：前大后小来自 QK 谱排序，而不是原始 hidden dimension
天然按编号排序。

待服务器轨迹可访问后，结果将写入：

`results/20260729_qksieve_band_distributions_multimodel_32k/`

其中：

- `band_detail.csv`：逐模型、主题、层、KV head、band 的完整结果；
- `band_aggregate.csv`：8 个 band 的聚合分布；
- `summary.json`：单调率、band 8 / band 1 比率和前 3 band 占比；
- `band_distributions.png`：K RMS、held-out Q RMS 和 score 贡献曲线。
