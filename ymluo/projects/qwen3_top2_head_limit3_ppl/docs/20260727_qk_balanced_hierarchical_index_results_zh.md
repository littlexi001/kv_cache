# QK-Balanced 层级自索引：当前方法、理论与实验记录

更新时间：2026-07-27

## 1. 当前结论

当前最好的可运行方法是：

> Query-conditioned QK-balanced biorthogonal transform  
> + 每层、每 KV head 的物理成本感知层级量化  
> + packed variable-bit Key index  
> + sampled-quantile 直接候选消费  
> + exact sparse QK/V attention

它不训练 router，不做 exact rerank，不回退 Full attention，也不使用任务标签。

在 Qwen3-4B、128K 历史、4 个完全留出的 256-token 窗口上：

| 指标 | 结果 |
|---|---:|
| Full PPL | 12.8848 |
| 本方法 PPL | 12.9147 |
| 质量保持率（Full PPL / Ours PPL） | **99.7688%** |
| 最差单窗口保持率 | **98.7451%** |
| 实际 attention token/head | 1345.6 |
| 实际 attention 比例 | **1.0512%** |
| 压缩 K 索引 / Full K+V FP16 | **5.7933%** |
| 稳态 decode 加速 | **4.0581x** |
| 含固定建索引成本的 256-token 加速 | **3.1671x** |
| 估计 break-even | **23.46 个生成 token** |
| 1024-token 生成投影加速 | **3.7924x** |

配对层级 moving-block bootstrap（4 个窗口、1024 个 token、50,000 次）：

| 统计量 | 结果 |
|---|---:|
| 质量保持率点估计 | 99.7688% |
| 质量保持率 95% CI | **[98.6239%, 100.8792%]** |
| P(retention >= 95%) | **100.000%** |
| P(retention >= 98%) | **99.834%** |
| P(retention >= 99%) | 90.794% |

注意：这里的 5.79% 是新增的压缩 K 检索索引相对 Full K+V FP16
的逻辑存储比例，不是整个运行时的 GPU KV 占用。目前完整 K/V 仍驻留 GPU，
用于取出被选 token 并执行 exact sparse attention。

## 2. 为什么旧方法在 128K 失真

先在完全相同的 128K 窗口上拆分“预算不足”和“检索器不准”：

| 方法 | 候选比例 | 聚合质量保持率 | 最差窗口 |
|---|---:|---:|---:|
| 旧 QK-Metric CountCap | 约 1% | 86.501% | 73.301% |
| packed Key-PCA | 约 1% | 93.703% | 88.975% |
| packed Key-PCA | 约 2.1% | 97.384% | 94.571% |
| exact QK top-1% | 1% | **100.008%** | 96.004% |
| QK-balanced packed（开发窗口） | 约 1.05% | **100.551%** | **98.014%** |

关键诊断是：exact top-1% 已经足够，因此 128K 的主要问题不是 token
预算太小，而是低维或低比特代理分数没有正确保留 query-key 的排序。

Key-PCA 只最大化 Key 自身的重构能量。一个 Key 方差很大的方向，不一定是
当前 query 经常使用的方向；反之，一个 Key 方差较小的方向也可能对
QK score 很重要。因此，单独对 K 做 PCA 的目标与检索目标不一致。

## 3. QK-balanced 双正交坐标

### 3.1 定义

对一个 layer/KV head，令历史 Key 和当前 prompt 尾部 Query 的二阶矩为：

$$
C_k = E[kk^\top], \qquad C_q = E[qq^\top].
$$

Query 样本较少，所以实际实现使用固定 0.5 shrinkage：

$$
\widetilde C_q
= (1-\lambda)C_q
+ \lambda \frac{\operatorname{tr}(C_q)}{d}I,
\qquad \lambda=0.5.
$$

对下式做 SVD：

$$
\widetilde C_q^{1/2} C_k^{1/2}
= U\Sigma V^\top.
$$

构造 Query 与 Key 的两个不同变换：

$$
A
= \widetilde C_q^{-1/2}U\Sigma^{1/2},
\qquad
B
= C_k^{-1/2}V\Sigma^{1/2}.
$$

定义：

$$
q' = A^\top q,\qquad k'=B^\top k.
$$

### 3.2 满维变换不改变精确 QK score

满秩时：

$$
AB^\top = I.
$$

所以：

$$
{q'}^\top k'
=q^\top AB^\top k
=q^\top k.
$$

这与普通 PCA 降维不同：在量化前保留全部 128 维时，变换本身没有
QK score 误差。误差只来自后续按频带丢弃/量化。

同时：

$$
A^\top \widetilde C_q A = \Sigma,
\qquad
B^\top C_k B = \Sigma.
$$

因此 Query 与 Key 在新坐标中的二阶能量同时被对角化，并按对 QK score
的重要性排序。实验中前 16 个 QK-balanced 坐标平均包含 94.82% 的
score energy，前 48 个包含 97.99%；Key-PCA 前 16 维的 Key energy
只有 77.32%，且它不是直接的 score energy。

### 3.3 截断的最优性

在 Query 与历史 Key 使用乘积二阶矩的模型下，对任意秩不超过 $r$ 的
双线性近似 $P$：

$$
E[(q^\top(I-P)k)^2]
=
\left\|
\widetilde C_q^{1/2}(I-P)C_k^{1/2}
\right\|_F^2.
$$

取：

$$
P_r=A_rB_r^\top
$$

等价于保留
$\widetilde C_q^{1/2}C_k^{1/2}$ 的前 $r$ 个奇异分量。由
Eckart-Young 定理：

$$
\min_{\operatorname{rank}(P)\le r}
E[(q^\top(I-P)k)^2]
=
\sum_{j>r}\sigma_j^2.
$$

这说明 QK-balanced 是针对 score MSE 的低秩坐标，而不是针对 K
重构误差的坐标。该结论依赖二阶矩模型；实际 attention 的相关性和
softmax 非线性由后续真实 trace 与端到端 PPL 实验验证，不能仅靠定理替代。

## 4. 层级量化与自动 rate allocation

### 4.1 频带

每个 128 维 Key 被分成 8 个连续频带，每个频带 16 维：

$$
\mathcal G_g=\{16g,\ldots,16g+15\},\quad g=0,\ldots,7.
$$

每个频带可以选择：

$$
b_g\in\{0,1,2,4,8\}.
$$

`0-bit` 表示不存该频带；非零频带额外保存一个 FP16 scale。

### 4.2 直接优化 Query 加权 score distortion

对频带 $g$ 和候选 bit 数 $b$，使用 prompt 尾部 Query 直接计算：

$$
D_g(b)=
\frac{1}{|\mathcal Q||\mathcal K_s|}
\left\|
Q'_g
\left(K'_{s,g}-\widehat K'_{s,g}(b)\right)^\top
\right\|_F^2.
$$

其中 $\mathcal K_s$ 是每 32 个历史 token 取一个的 Key 样本。

随后用小型动态规划求：

$$
\min_{\{b_g\}}\sum_g D_g(b_g)
$$

满足物理成本约束：

$$
\sum_g
\left(
b_g+\mathbf 1[b_g>0]
\right)
\le 15.
$$

约束中的 `+1` 是每个 active band 的 FP16 scale 成本，避免出现
“逻辑 bit 数很小，但 scale metadata 很大”的虚假压缩。

每个 token/head 的索引 bit 数为：

$$
R_{\text{index}}
=16\sum_g b_g
+16\sum_g\mathbf 1[b_g>0].
$$

相对一个 FP16 K+V（$2\times128\times16=4096$ bit）：

$$
\rho_{\text{index}}=R_{\text{index}}/4096.
$$

跨 32K/96K 的 280 个 layer-head，自动分配最常见的结构为：

| allocation | layer-head 数 |
|---|---:|
| 4-4-4-0-0-0-0-0 | 88 |
| 8-4-0-0-0-0-0-0 | 72 |
| 8-1-1-1-0-0-0-0 | 57 |
| 4-4-2-1-0-0-0-0 | 33 |
| 4-4-1-2-0-0-0-0 | 18 |
| 其他 | 12 |

这不是手工规定“前 16 维一定 8-bit”，而是每层、每 head 根据当前
Q/K 数值统计自行选择。

## 5. 检索和 attention 执行

### 5.1 Query 校准

1. 正常执行 dense prefill。
2. 保存 prompt 尾部 8 个 token 的 Query。
3. GQA 下把同一 KV head 对应的多个 Query head 合并，因此 Qwen3-4B
   每 KV head 实际有 $8\times4=32$ 个 Query 向量。
4. 使用这些 Query 估计 $\widetilde C_q$，不需要训练数据或任务标签。

### 5.2 候选预算

目标 token 数：

$$
B(N)=
\min\left(
N,\ 1280,\ \max(256,\lceil0.06N\rceil)
\right).
$$

对应：

| 历史长度 | 目标 token/head | 目标比例 |
|---:|---:|---:|
| 2K | 256 | 12.50% |
| 4K | 256 | 6.25% |
| 8K | 492 | 6.00% |
| 16K | 960 | 6.00% |
| 24K | 1280 | 5.33% |
| 32K | 1280 | 4.00% |
| 64K | 1280 | 2.00% |
| 128K | 1280 | 1.00% |

### 5.3 sampled-quantile 单遍扫描

1. 把变换后的 Query 量化为 INT8。
2. 从压缩索引采样：

$$
S=\min\left(2048,\max\left(256,\left\lceil16/p\right\rceil\right)\right),
$$

   其中 $p=B(N)/N$。
3. 用采样分数估计保留比例 $p$ 的阈值。
4. CUDA kernel 扫描 packed variable-bit index，直接输出超过阈值的 token。
5. 不对候选再做 exact QK top-k 重排。
6. 对输出 token 从完整 K/V 取值，执行 exact sparse QK、softmax 和 V 聚合。

128K 留出集目标为 1280 token，实际平均输出 1345.6 token（1.0512%），
候选 buffer overflow 为 0。

## 6. 跨模型与跨领域检索结果

数据：

- Qwen3-4B：sports、medicine，32K 与 96K；
- Llama-3.1-8B：sports、medicine，32K；
- Qwen2.5-7B：sports、medicine，32K；
- 每个 32K trace 前 8 步校准、后续步骤测试；
- 新采 96K trace 前 8 个 query 校准、后 24 个 query 测试；
- Key-PCA 与 QK-balanced 使用相同约 5.75% 索引成本。

加权汇总：

| 方法 | Index / Full KV | 真实 top-1% recall | 被选 attention mass | Oracle top-1% mass recall | Score Pearson | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Key-PCA | 5.772% | 59.643% | 72.197% | 90.665% | 0.9394 | 1.0848 |
| QK-balanced | **5.745%** | **70.635%** | **76.548%** | **96.423%** | **0.9643** | **0.8186** |

六个 32K model-topic 组合中，QK-balanced 的 top-1% recall、
attention-mass recall、Pearson 和 RMSE 均优于 Key-PCA，没有反例。

96K 独立 query：

| 数据 | 方法 | top-1% recall | Oracle mass recall | Pearson | RMSE |
|---|---|---:|---:|---:|---:|
| sports 96K | Key-PCA | 62.048% | 88.041% | 0.9674 | 1.0706 |
| sports 96K | QK-balanced | **72.531%** | **97.040%** | **0.9841** | **0.7938** |
| medicine 96K | Key-PCA | 60.815% | 88.829% | 0.9634 | 1.0871 |
| medicine 96K | QK-balanced | **70.794%** | **96.639%** | **0.9811** | **0.8211** |

## 7. 固定位宽消融

在三个模型、两个领域的 32K trace 上：

| QK 坐标内的分配 | Index / Full KV | top-1% recall | Oracle mass recall | Pearson | RMSE |
|---|---:|---:|---:|---:|---:|
| 固定 4-4-4 | 5.859% | 69.133% | 94.109% | 0.9540 | 0.9590 |
| 固定 8-2-2 | 5.859% | 66.542% | 95.762% | 0.9506 | 0.9360 |
| 固定 8-4-0 | 5.469% | 67.012% | 95.691% | 0.9503 | 0.9005 |
| **自动 qMSE，budget=15** | **5.747%** | **70.482%** | **96.361%** | **0.9616** | **0.8203** |
| 固定 8-4-4 | 7.422% | 74.618% | 97.439% | 0.9710 | 0.6837 |

结论：

1. 用户提出的 8/4 层级精度方向有效；固定 8-4-4 的代理分数最好。
2. 它多用了约 29% 的索引空间，且扫描计算更多。
3. 相近物理成本下，自动 qMSE 分配优于固定 4-4-4、8-2-2 和 8-4-0。
4. 因此当前默认保留 budget=15 自动方案；8-4-4 作为质量上界消融。
5. 低于 1-bit 的 tail 融合此前 CountSketch 类实验没有得到稳定收益，
   当前不进入主方法。

## 8. 128K 端到端稳定性

完全留出的 1024 个 teacher-forced token：

| 指标 | 结果 |
|---|---:|
| Full / Sparse top-1 一致率 | 96.0938% |
| top-1 翻转率 | 3.9062% |
| KL(Full || Sparse) | 0.01032 |
| JS divergence | 0.00257 |
| 目标 token 平均 NLL 增量 | 0.002315 |
| 基于 margin 的充分证书通过率 | 53.03% |

证书通过率低于真实 top-1 一致率是正常的：证书是保守充分条件，
“未通过”不等于预测一定翻转。

## 9. 与近期工作的边界

不能声称：

- 首个低 bit Key 检索；
- 首个 token-level KV retrieval；
- 首个 self-indexing KV cache；
- 首个 mixed-precision KV 方法；
- 首个 PCA/SVD KV 压缩。

相关工作已经覆盖这些单点：

- FIER：统一 1-bit Key 的 token-level 检索，约 11% cache budget，
  报告 1.2x–1.5x decode 加速；
- Self-Indexing KVCache：sign-based 1-bit VQ Key 同时作为索引；
- RaBitQCache：随机旋转 binary Key、INT4 Query、无偏 score estimator
  与 adaptive top-p；
- RateQuant：基于 rate-distortion 的跨 head / quantizer mixed precision；
- SVDq/Palu：Key 或 KV latent channel 的低秩物理压缩。

当前可辩护的差异是：

1. **Query-conditioned biorthogonal coordinates**：不是只压 K，而是联合
   $C_q,C_k$，满维变换在量化前严格保持 $q^\top k$。
2. **Score-optimal spectrum**：秩 $r$ 截断最小化二阶矩模型下的双线性
   score MSE，残差可写成尾部奇异值平方和。
3. **Within-head spectral rate allocation**：在每个 layer/head 内，对
   8 个频带使用 0/1/2/4/8-bit，并显式计入 scale metadata。
4. **No learned router / no fallback / no rerank**：由当前 prompt 的数值统计
   自校准，候选直接进入 exact sparse attention。
5. **同预算实证闭环**：Key-PCA 在约 5.77% 索引下明显弱于
   QK-balanced，而 exact top-1% 证明 token budget 本身足够。

主要参考：

- FIER: https://aclanthology.org/2025.findings-emnlp.515/
- Self-Indexing KVCache: https://ojs.aaai.org/index.php/AAAI/article/view/39988
- RaBitQCache: https://arxiv.org/abs/2606.31519
- RateQuant: https://arxiv.org/abs/2605.06675

## 10. 当前尚未解决的问题

1. 新 QK-balanced 方法的完整 16-task LongBench 主表尚未完成；
   当前只有真实生成 smoke 和 128K PPL。
2. Llama-3.1 与 Qwen2.5 目前是 QK trace 泛化验证，尚缺新方法的
   端到端 PPL/生成质量主表。
3. RULER 64K/128K 需要重新用新方法跑，不能沿用旧 CountCap 分数。
4. 完整 K/V 仍驻留 GPU；当前工作证明 attention 计算/读取缩减与
   低成本检索，但还不是完整物理 KV 显存压缩系统。
5. 8K/16K 下 fixed overhead 仍可能抵消稀疏收益；应明确报告
   break-even，而不是只报 128K 最优速度。
6. Query covariance shrinkage 固定为 0.5，需要补敏感性实验或推导
   可复现的无调参 shrinkage。
7. 二阶矩最优性依赖乘积矩模型，需要在论文里将其写成明确假设，
   不应包装成对真实 softmax attention 的无条件定理。
8. 还缺与 FIER、Self-Indexing、RaBitQCache 在同模型、同硬件、
   同候选比例下的直接复现。

因此，这个结果显著增强了方法创新性与质量，但目前仍不能诚实地称为
“稳定 8/10 的 ICLR 投稿”。达到该目标的关键不再是继续调 bit 参数，
而是完成独立 benchmark、强 baseline 和物理系统闭环。

## 11. 代码位置

核心实现：

- `src/run_head_top2_targeted_ppl_20260714.py`
  - `_packed_qmse_initialize`
  - `_packed_qmse_rebuild_index`
  - `_packed_qmse_spectral_attention`
  - score mode:
    `pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct`
- `src/variablebit_spectral_cuda_20260727.py`
  - variable-bit packed index
  - Key encode
  - INT8 Query
  - sampled threshold scan
- `src/run_direct_countcap_denseprompt_ppl_20260725.py`
  - 128K Full / Sparse paired PPL harness
- `src/analyze_qk_balanced_spectral_rate_20260727.py`
  - Key-PCA / QK-balanced / fixed-rate trace comparison
- `src/bootstrap_qkmetric_128k_holdout_20260727.py`
  - paired hierarchical moving-block bootstrap
- `src/run_sample_calibrated_longbench_20260717.py`
  - LongBench generation integration

启动脚本：

- `scripts/launch_qkmetric_packed_128k_8gpu_20260727.sh`
- `scripts/launch_qkmetric_packed_128k_holdout_8gpu_20260727.sh`
- `scripts/launch_qk_balanced_spectral_rate_crossmodel_8gpu_20260727.sh`
- `scripts/launch_qk_balanced_96k_trace_and_analysis_2gpu_20260727.sh`
- `scripts/launch_qk_balanced_fixed_rate_ablation_4gpu_20260727.sh`
- `scripts/launch_qkbalanced_longbench_m2_8gpu_20260727.sh`

## 12. 结果位置

- 128K 开发：
  `results/20260727_qkmetric_packed_128k`
- 128K 独立留出：
  `results/20260727_qkmetric_packed_128k_holdout`
- 128K 合并结果：
  `results/20260727_qkmetric_packed_128k_combined_analysis`
- bootstrap：
  `results/20260727_qkmetric_packed_128k_combined_analysis/bootstrap_block16`
- 32K 跨模型 trace：
  `results/20260727_qk_balanced_spectral_rate_crossmodel`
- 96K 独立 query trace：
  `results/20260727_qk_balanced_96k_independent`
- 32K+96K 汇总：
  `results/20260727_qk_balanced_crossmodel_combined_32k96k`
- 固定位宽消融：
  `results/20260727_qk_balanced_fixed_rate_ablation`
- LongBench 生成 smoke：
  `results/20260727_qkbalanced_longbench_smoke`

## 13. 最小复现命令

服务器项目目录：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
export PATH=/home/fdong/miniconda3/envs/moe/bin:/usr/local/cuda/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

运行 128K 独立留出：

```bash
bash scripts/launch_qkmetric_packed_128k_holdout_8gpu_20260727.sh
```

运行跨模型 trace：

```bash
bash scripts/launch_qk_balanced_spectral_rate_crossmodel_8gpu_20260727.sh
bash scripts/launch_qk_balanced_96k_trace_and_analysis_2gpu_20260727.sh
```

运行固定层级量化消融：

```bash
bash scripts/launch_qk_balanced_fixed_rate_ablation_4gpu_20260727.sh
```

运行 LongBench m2 真实生成 smoke：

```bash
bash scripts/launch_qkbalanced_longbench_m2_8gpu_20260727.sh
```

运行 bootstrap：

```bash
/home/fdong/miniconda3/envs/moe/bin/python \
  src/bootstrap_qkmetric_128k_holdout_20260727.py \
  --input_root results/20260727_qkmetric_packed_128k_holdout \
  --output_dir \
    results/20260727_qkmetric_packed_128k_combined_analysis/bootstrap_block16 \
  --block_length 16 \
  --replicates 50000
```
