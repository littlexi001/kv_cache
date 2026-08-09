# QK-Balanced 层级量化索引：方法、理论、系统与实验结论

更新时间：2026-07-27

## 1. 结论

当前可以冻结的是**方法骨架**，还不能冻结 auto/fixed 位宽和 scale 估计的最终
端点：

> QK-balanced 双正交坐标  
> + 每层、每 KV head 的 score-MSE 层级位宽分配  
> + 约 240 bit/token/head 的 packed Key 检索索引  
> + packed full-topk 或 sampled-quantile 候选输出  
> + 候选上的精确 Q/K/V sparse attention  
> + group16 INT4 临时 prefill KV 与直接 Key 转码

它不训练 router，不使用任务标签，不做 exact rerank，不回退 Full
attention。当前实现的候选预算为：

$$
B(N)=\min\{N,1280,\max(256,\lceil0.06N\rceil)\}.
$$

在 Qwen3-4B、128K 历史、4 个独立窗口、共 1024 个 teacher-forced
token 上：

| 指标 | 当前结果 |
|---|---:|
| Full PPL | 12.88480 |
| Ours PPL | 12.89227 |
| 质量保持率 | **99.9421%** |
| 质量保持率 95% CI | **[98.8572%, 101.0359%]** |
| 实际 attention token/head | 1345.51 |
| 实际 attention 比例 | **1.0512%** |
| 索引 / Full FP16 K+V | **5.8577%** |
| Full 整模型 decode | 297.70 ms/token |
| Ours 整模型稳态 decode | 69.03 ms/token |
| 整模型稳态加速 | **4.3129x** |
| 建索引固定成本 | 3.476 s |
| 256-token 实测加速，含固定成本 | **3.6017x** |
| break-even 生成长度 | **15.20 token** |

这是当前 qMSE 自动分配 + raw Q-aware scale 的最佳 128K **逻辑路径**
结果。它不能单独决定最终方法，因为首个 16-task LongBench 开发切分上的
allocation $\times$ scale 因子实验已经得到：

| 端点 | Macro score | 相对 Full |
|---|---:|---:|
| Full | 0.453665 | 100.000% |
| auto + plain scale | 0.453245 | **99.907%** |
| fixed `4-4-2-1` + plain scale | 0.453345 | **99.929%** |
| auto + raw Q-aware scale | 0.452185 | 99.674% |
| fixed `4-4-2-1` + raw Q-aware scale | 0.452675 | 99.782% |
| auto + OAS Q-aware scale | 0.452392 | 99.719% |
| fixed `4-4-2-1` + OAS Q-aware scale | 0.452649 | 99.776% |

fixed 相对 auto 的差异很小且 bootstrap 区间跨零；raw Q-aware scale 在短而
异质的 LongBench 上没有复现 128K PPL 收益。当前稳健参考因此改为
`auto + plain scale`，固定 `4-4-2-1` 是同码率强对照，raw Q-aware scale
降为 128K 专用消融。OAS 二阶矩收缩相对 raw auto 恢复了 0.000207 macro，
但仍比 plain auto 低 0.000853；两项 bootstrap 区间均跨零。OAS 因而也不
替换 plain scale。offset-100 独立切分和正式协议七路消融继续运行，用于验证
这一负结论，而不是在测试集上重新选择方法。

完整 LongBench 质量实验也已完成：

| 指标 | Llama-3.1-8B-Instruct |
|---|---:|
| 英文任务数 | 16 |
| Full/Ours 严格配对 | 3750 |
| Full macro score | 0.458157 |
| Ours macro score | 0.457260 |
| macro 差值 | -0.000897 |
| 质量保持率 | **99.8042%** |
| 分层 paired-bootstrap 95% CI | **[99.1023%, 100.4304%]** |
| $P(\text{保持率}\ge98\%)$ | **100%** |
| Python 质量 harness online speed | **0.5358x** |

分任务结果为：

| 任务 | Full | Ours | 相对 Full |
|---|---:|---:|---:|
| 2WikiMQA | 0.46696 | 0.46934 | 100.51% |
| GovReport | 0.33618 | 0.33670 | 100.15% |
| HotpotQA | 0.48113 | 0.48246 | 100.28% |
| LCC | 0.63299 | 0.63287 | 99.98% |
| MultiNews | 0.27006 | 0.26836 | 99.37% |
| MultiFieldQA-en | 0.56482 | 0.56256 | 99.60% |
| Musique | 0.27549 | 0.28023 | 101.72% |
| NarrativeQA | 0.24350 | 0.23870 | 98.03% |
| PassageCount | 0.08200 | 0.07875 | 96.04% |
| PassageRetrieval-en | 0.69000 | 0.68500 | 99.28% |
| Qasper | 0.45504 | 0.44131 | 96.98% |
| QMSum | 0.22954 | 0.23596 | 102.80% |
| RepoBench-P | 0.54369 | 0.54657 | 100.53% |
| SAMSum | 0.44515 | 0.44340 | 99.61% |
| TREC | 0.69000 | 0.69000 | 100.00% |
| TriviaQA | 0.92395 | 0.92395 | 100.00% |

该 LongBench 结果对应 plain QK-balanced 自动分配版本，不是上表的 Q-aware
scale 版本。LongBench 平均 prompt 约 5.9K，
索引构建与逐层 Python 调度无法摊销，所以 0.5358x 不能替代 32K--128K
物理缓存和 CUDA kernel 速度结论。

还必须区分 prompt 协议。上述 3750 对使用“完整保留 task/query suffix，只对
context 做中间截断”的 question-aware 协议。token-level 审计发现，它与
KVCache-Factory 官方“完整 prompt 先取首尾各半、再包 chat template”的最终
token 序列只有 **23.6%** 完全一致；3750 条中有 2097 条原始 prompt 超过
7500 token。因此，上述结果不能直接标成 AdaKV Table 5 的完全同协议复现。
严格 `official_middle` 模式已经实现，并已排队运行 Full + 六个
allocation $\times$ scale 稀疏端点的 3750 条七路完整实验；正式论文主表
以该结果为准。

32K 物理 KV 路径也已完成四主题、每主题 64 个 teacher-forced token 的严格
配对。这里把远端完整 K/V 放在 pinned CPU memory，GPU 只保留压缩 Key
索引、小型 exact hot cache 和运行时缓冲：

| 物理路径 | 聚合 PPL | 质量保持率 | GPU KV 比例 | 在线整模型加速 |
|---|---:|---:|---:|---:|
| Full GPU KV | 14.97299 | 100.000% | 100.000% | 1.000x |
| PCA64 + full-topk | 16.33662 | 91.653% | 10.883% | 3.101x |
| QK-balanced + full-topk | **15.14058** | **98.893%** | **10.451%** | **3.509x** |
| QK-balanced + sampled-compact | 15.15921 | 98.772% | 10.477% | 3.090x |

QK-balanced full-topk 的质量保持率 bootstrap 95% CI 为
`[96.321%, 101.678%]`。其平均额外 prefill+建索引成本为 43.62 秒，按本批
在线 token 延迟计算，聚合 break-even 为约 120 token。sampled-compact 在
32K 既更慢又略差，暂不作为短序列后端。

128K 四窗口物理实验也已完成。每个端点严格配对 1024 个 teacher-forced
token，远端 exact K/V 位于 pinned CPU memory：

| 128K 物理路径 | PPL | 质量保持率 | GPU KV 比例 | token/head | 在线加速 |
|---|---:|---:|---:|---:|---:|
| Full GPU KV | 12.88480 | 100.000% | 100.000% | 128K | 1.000x |
| auto + full-topk | 12.90005 | 99.882% | 9.299% | 1280.0 | **2.242x** |
| auto + sampled-compact | **12.88955** | **99.963%** | 9.315% | 1337.9 | 1.851x |
| fixed `4-4-2-1` + sampled-compact | 12.89094 | 99.952% | 9.316% | 1337.8 | 2.102x |

这里的“在线加速”包含每个 decode step 的检索、候选消费、CPU/GPU exact
cache 访问和 attention，但不包含首次物理 cache 构造。当前 sampled-compact
直到 128K 仍未快过 full-topk，说明它减少扫描读流量的收益被采样、溢出处理
和不规则候选消费开销抵消。

最初的物理 cache 构造很慢。以 Full dense prompt prefill 作为公平固定成本
基线，旧 CPU exact-K/V offload 的四窗口聚合结果为：

| 128K 物理路径 | 1024-token 总加速 | 估计 break-even |
|---|---:|---:|
| auto + full-topk | 0.620x | 5394 token |
| auto + sampled-compact | 0.616x | 6387 token |
| fixed `4-4-2-1` + sampled-compact | **0.651x** | **5062 token** |

定位后发现，旧实现每个 prefill chunk 都把此前所有 exact K/V 从 CPU 重新
载入 GPU，产生近二次 PCIe 流量；转换阶段又上传 K/V 并回写一份重复 host
cache。新实现采用两步数值与系统闭环：

1. prefill 时把新产生的 exact K/V 只写入 pinned CPU 一次，同时在 GPU 保留
   group16 INT4 临时历史；每 16 维使用一个 FP16 scale，有效成本为
   $4+16/16=5$ bit/coordinate；
2. query 到达并确定 QK-balanced 坐标后，直接从 GPU 临时 INT4 K 构造最终
   packed Key 索引，并复用原 pinned-CPU K/V，不再上传完整 V 或回写重复 K/V。

在 Qwen3-4B 的同一 128K `mixed_a_w2` 窗口上：

| prefill/转换路径 | PPL | 相对 Full | 固定阶段 | 在线加速 | 256-token 总加速 | 峰值 GPU |
|---|---:|---:|---:|---:|---:|---:|
| Full GPU KV | 10.54340 | 100.000% | 217.43 s | 1.000x | 1.000x | - |
| 四卡常驻 prefill | 10.65899 | 98.916% | 238.66 s | 1.973x | 1.058x | 9.92--11.38 GiB/卡 |
| 单尺度 INT4 临时 KV | 11.63026 | 90.655% | 214.85 s | 2.241x | 1.179x | 17.68 GiB |
| group16 INT4，exact-host K 建索引 | 10.65193 | 98.981% | 223.89 s | 2.130x | 1.130x | 18.65 GiB |
| **group16 INT4，直接 K 转码** | **10.64367** | **99.058%** | **191.13 s** | **2.229x** | **1.302x** | **18.65 GiB** |

直接转码的最终物理 GPU KV 比例仍为 9.316%；18.65 GiB 是 prefill 临时峰值，
不能与转换后的常驻比例混用。其转换时间由 33.57 秒降到 3.56 秒，固定阶段
已经低于 Full，因此该窗口的估计 break-even 为 0 token。该结论目前是一个
128K 窗口的系统闭环证据，仍需按四窗口复测后进入主表。

32K 的四主题复核给出更稳健的质量证据。体育、医学、计算机和宗教共 256 个
严格配对 token 上，group16 prefill 相对 exact prefill 保持 **99.827%**，
paired-bootstrap 95% CI 为 `[98.669%, 101.110%]`；加入直接 K 转码后为
**99.689%**，95% CI 为 `[98.439%, 101.121%]`。直接转码的平均转换时间为
3.21 秒、固定阶段为 23.70 秒。

机制实验在 36 层、每层 1000 个历史 token 和 9216 个
layer/head/query 单元上解释了单尺度 INT4 为何失败：

| 临时 KV 格式 | 有效 bit/coordinate | Key NMSE | score NMSE | top-1% recall | exact attention mass | 输出相对 MSE |
|---|---:|---:|---:|---:|---:|---:|
| INT4，每 token/head 一个 scale | 4.125 | 4.589% | 8.500% | 64.82% | 88.65% | 30.60% |
| **INT4，每 16 维一个 scale** | 5.000 | **0.714%** | **0.857%** | **82.83%** | **92.05%** | **4.81%** |

group16 把 score NMSE 降低约 9.9 倍、attention 输出误差降低约 6.4 倍。
非对称诊断进一步表明 K 是主要来源：K-group16/V-per-head 仍有 82.83% recall，
而 K-per-head/V-group16 只有 64.82%；但前者的输出误差仍比 K/V 都用 group16
高 63%，所以当前稳健系统配置仍保留 K/V 双 group16。

必须同时说明：

1. 5.86% 是**额外压缩 Key 索引**相对 Full FP16 K+V 的逻辑比例。
2. 128K 逻辑实验中的完整 FP16 K/V 仍驻留 GPU，所以 5.86% 只表示附加
   索引；32K/128K 物理实验则已把远端 K/V 放到 CPU，其约 10.45%/9.30%
   是实际 GPU KV 占用比例。两类数字不能混用。
3. 128K PPL 数据来自 20 Newsgroups 混合流；LongBench 是独立生成质量证据，
   RULER 仍在队列中。
4. 当前证据足以确定方法方向，但还不足以保证 ICLR 8/10。

## 2. 问题诊断

在相同 128K 窗口和约 1% attention token 预算下：

| 检索方式 | 质量保持率 |
|---|---:|
| 旧 QK-Metric CountCap | 86.50% |
| packed Key-PCA | 93.70% |
| exact QK top-1% | 100.01% |

因此，128K 失真的主要原因不是“只保留 1% token”，而是代理分数没有准确
保留真实 QK 排序。Key-PCA 优化的是 Key 重构：

$$
\min_P E\|k-Pk\|_2^2,
$$

而检索真正需要优化的是双线性 score：

$$
\min_P E\left(q^\top k-q^\top Pk\right)^2.
$$

Query 与 Key 的主方向并不完全相同，所以两者目标一般不等价。

## 3. QK-balanced 双正交坐标

### 3.1 定义

对每个 layer/KV head，用均匀采样的历史 Key 与 prompt 尾部 8 个位置的
Query 估计：

$$
C_k=E[kk^\top],\qquad C_q=E[qq^\top].
$$

这里的 $C_k,C_q$ 是不减均值的二阶矩，而不是中心化协方差；这与原始
$q^\top k$ 点积目标一致。

Query 样本少，因此使用固定 shrinkage：

$$
\widetilde C_q
=(1-\lambda)C_q
+\lambda\frac{\operatorname{tr}(C_q)}{d}I,
\qquad \lambda=0.75.
$$

令：

$$
\widetilde C_q^{1/2}C_k^{1/2}=U\Sigma V^\top.
$$

构造：

$$
A=\widetilde C_q^{-1/2}U\Sigma^{1/2},
\qquad
B=C_k^{-1/2}V\Sigma^{1/2}.
$$

Query 与 Key 使用不同坐标：

$$
q'=A^\top q,\qquad k'=B^\top k.
$$

### 3.2 满维 score 精确不变与协方差平衡

满秩时：

$$
AB^\top=I.
$$

所以：

$$
{q'}^\top k'=q^\top AB^\top k=q^\top k.
$$

变换本身不是近似。误差只来自后续丢弃坐标和低比特量化。

同一组变换还同时满足：

$$
A^\top\widetilde C_qA=\Sigma,
\qquad
B^\top C_kB=\Sigma.
$$

因此，QK-balanced 坐标既完整保留原始 QK 点积，又把正则化 Query
协方差和 Key 协方差变成同一个对角矩阵。前者保证不量化时没有变换误差；
后者使后续的逐频带 score 失真可以严格相加。

### 3.3 rank-r score-MSE 最优性

在 Query 与 Key 的乘积二阶矩模型下，任意 rank 不超过 $r$ 的双线性近似
$P$ 满足：

$$
E[(q^\top(I-P)k)^2]
=
\left\|
\widetilde C_q^{1/2}(I-P)C_k^{1/2}
\right\|_F^2.
$$

保留前 $r$ 个 QK-balanced 分量得到：

$$
P_r=A_rB_r^\top.
$$

由 Eckart--Young 定理：

$$
\min_{\operatorname{rank}(P)\le r}
E[(q^\top(I-P)k)^2]
=\sum_{j>r}\sigma_j^2.
$$

这给出的是在明确二阶矩假设下的 score-MSE 最优低秩近似，不是对任意未来
Query 或真实 softmax 输出的无条件保证。

## 4. 层级量化与物理成本分配

### 4.1 频带与位宽

128 个 QK-balanced 坐标分为 8 个连续频带，每带 16 维：

$$
\mathcal G_g=\{16g,\ldots,16g+15\}.
$$

每个频带独立选择：

$$
b_g\in\{0,1,2,4,8\}.
$$

`0-bit` 表示丢弃。每个非零频带额外保存一个 FP16 scale。

### 4.2 Query 加权失真

对频带 $g$、位宽 $b$，在采样 Key 与当前 prompt 的校准 Query 上计算：

$$
D_g(b)=
\frac{1}{|\mathcal Q||\mathcal K_s|}
\left\|
Q'_g(K'_{s,g}-\widehat K'_{s,g}(b))^\top
\right\|_F^2.
$$

随后动态规划求：

$$
\min_{\{b_g\}}\sum_gD_g(b_g)
$$

满足：

$$
\sum_g\left(b_g+\mathbf1[b_g>0]\right)\le15.
$$

每个 rate unit 对应 16 bit，因此物理索引成本为：

$$
R=16\sum_gb_g+16\sum_g\mathbf1[b_g>0]\le240\ \text{bit}.
$$

相对一个 128 维 FP16 K+V：

$$
\rho=R/(2\times128\times16)\approx5.86\%.
$$

实际平均为 5.80%，因为个别 layer/head 没有用满预算。

### 4.3 为什么逐频带目标有理论意义

对某个 Key 的量化残差记为：

$$
\varepsilon
=k'-\widehat k'
=(\varepsilon_0,\ldots,\varepsilon_{G-1}).
$$

在协方差为 $\widetilde C_q$ 的正则化 Query 二阶模型下：

$$
\begin{aligned}
\mathbb E_q\!\left[
\left({q'}^\top\varepsilon\right)^2
\right]
&=
\varepsilon^\top
\left(A^\top\widetilde C_qA\right)
\varepsilon\\
&=
\varepsilon^\top\Sigma\varepsilon\\
&=
\sum_g
\varepsilon_g^\top\Sigma_g\varepsilon_g.
\end{aligned}
$$

最后一个等号来自 $\Sigma$ 为对角矩阵。这里不需要假设不同频带的量化残差
相互独立，也不需要假设量化噪声为白噪声。若量化尺度和位宽分配都使用该
正则化对角度量，对采样 Key 再取平均后，恰好得到：

$$
D(\mathbf b)=\sum_gD_g(b_g).
$$

所以，对独立逐频带量化器、给定候选位宽集合和总 bit 预算，动态规划求得的是
正则化二阶模型下的全局最优离散分配，不只是一个 Cauchy--Schwarz 上界。

当前生产候选使用的是 prompt 尾部 8 个 Query 的**经验逐频带 qMSE**：
它保留每个 16 维频带内部的完整协方差，但在 DP 中不显式加入跨频带项。
因此，DP 对“经验频带 qMSE 之和”精确最优，却不自动等价于完整经验 score
qMSE。新增的同码率对照使用 $\Sigma$ 构造严格对角度量，称为
`regularized-diagonal qMSE`；它与上述定理完全一致，也减少小样本校准噪声。
最终冻结哪一个版本由 held-out 结果决定。

这个结论仍是条件定理，而不是对任意未来 Query 的无条件保证。shrinkage
后的 $\widetilde C_q$ 与真实 held-out Query 协方差可能不同，因此实验会额外
报告：

1. 变换后正则化 Query 与采样 Key 协方差的非对角能量；
2. held-out Query 协方差的非对角能量；
3. 实际总 score qMSE 与各频带 qMSE 之和的交叉项；
4. 校准 qMSE 对 held-out qMSE 的相关性。

32K sports/medicine 的 7680 个 held-out query 实验已闭环：

| 量 | sports | medicine |
|---|---:|---:|
| 正则化 Query 非对角能量比 | $2.95\times10^{-6}$ | $2.97\times10^{-6}$ |
| 采样 Key 非对角能量比 | $2.32\times10^{-5}$ | $2.17\times10^{-5}$ |
| held-out Query 非对角能量比 | 0.557 | 0.573 |
| $|D_{\rm cross}|/D_{\rm additive}$ 均值 | 4.46% | 4.21% |
| $D_{\rm actual}/D_{\rm additive}$ 均值 | 1.020 | 1.008 |
| 上述比值 p90 | 1.097 | 1.077 |
| log 校准误差与 log held-out 误差 Pearson | 0.491 | 0.402 |

这验证了两个不同层次的结论。第一，逐频带可加性在构造使用的正则化二阶模型
上数值精确；第二，单个未来 Query 并不服从该对角协方差，实际交叉项虽平均
较小，却不是零，而且校准误差对 held-out 难度只有中等预测力。这也解释了
为什么自动逐 head 分配目前没有稳定超过固定 `4-4-2-1`。论文只能声称
“模型内精确 + held-out 近似成立”，不能声称任意 Query 上严格可加。

### 4.4 小样本 Query 度量的 OAS 收缩

raw Q-aware scale 直接使用少量 prompt-tail Query 的经验二阶矩，容易把当前
问题末尾的偶然方向放大。新候选对投影 Query $q'$ 的未中心化二阶矩

$$
S=\frac1n\sum_{j=1}^n q'_j{q'_j}^{\top}
$$

做闭式 OAS 收缩：

$$
\alpha
=\operatorname{clip}_{[0,1]}
\frac{(1-2/d)\|S\|_F^2+\operatorname{tr}(S)^2}
{(n+1-2/d)\left(\|S\|_F^2-\operatorname{tr}(S)^2/d\right)},
$$

$$
\widehat S
=(1-\alpha)S
+\alpha\frac{\operatorname{tr}(S)}dI.
$$

当 Query 少或经验方向不稳定时，$\alpha$ 接近 1；样本增多且各向异性证据
稳定时，$\alpha$ 自动降低。它没有训练参数、任务标签或验证集拟合。位宽
分配使用 $\widehat S$ 计算 qMSE；对某个 token、频带的整数 code $c$，
量化 scale 使用同一度量下的闭式解：

$$
a^*
=\max\left(0,\frac{c^\top\widehat S_g k}
{c^\top\widehat S_g c}\right).
$$

因此 OAS 版本不是学习一个新模块，而是把 noisy plug-in metric 替换为
小样本风险更低的二阶矩估计。但开发切分上它只部分修复 raw Q-aware scale，
仍低于 plain scale，故不进入通用主方法。

## 5. 候选检索与执行

### 5.1 索引构建

1. 正常执行 dense prefill。
2. 均匀采样历史 Key，估计 $C_k$。
3. 收集 prompt 尾部 8 个 Query；GQA 下同一 KV head 的 Query heads 合并。
4. 估计 $\widetilde C_q$，构造 $A,B$。
5. 对每个 layer/head 运行物理成本约束下的 qMSE 位宽分配。
6. 把全部历史 Key 编码为 packed variable-bit index。

### 5.2 单个 decode 位置

先用融合 CUDA kernel 完成 $q'=A^\top q$ 后的逐频带 INT8 Query 量化。
候选枚举有两个后端，二者使用相同的 packed proxy score，也都不做 exact
rerank：

- `full-topk`：扫描完整压缩索引并直接取 proxy top-$B$；32K 实测更快；
- `sampled-compact`：先采样估计阈值，再单遍 compact 超阈值项；用于检验
  很长序列下减少全局 top-k 开销是否有收益。

sampled-compact 的采样数为：

$$
S=\min\left(2048,\max\left(256,\left\lceil16/p\right\rceil\right)\right),
\qquad p=B(N)/N.
$$

随后用样本估计代理分数的 $(1-p)$ 分位数，单次扫描 packed index 并 compact
所有超过阈值的 token。最终从 exact K/V gather 候选，在候选上重算精确
QK、softmax 和 V 聚合。

128K 目标为 1280 token/head，阈值误差使实际均值为 1345.42，候选 overflow
率为 $8.51\times10^{-7}$。

## 6. 从代理误差到 attention 输出

### 6.1 top-k 充分条件

设真实第 $k$ 与第 $k+1$ 个分数间隔为：

$$
\gamma_k=s_{(k)}-s_{(k+1)}.
$$

若所有代理误差满足：

$$
\|\widehat s-s\|_\infty<\gamma_k/2,
$$

则代理 top-k 集合与真实 top-k 集合完全相同。这是保守充分条件；不满足条件
并不表示候选一定错误。

### 6.2 遗漏 attention mass 的直接界

最终 attention 使用候选上的**原始精确 logits**，所以代理 score 的绝对
误差不会继续进入 softmax；它只决定候选集合。设 Full attention 在候选外
遗漏的概率质量为：

$$
\eta=\sum_{i\notin S}p_i.
$$

把候选内概率重新归一化后，严格有：

$$
\|p-\widetilde p\|_1=2\eta.
$$

若 attention 输出为 $o=\sum_ip_iv_i$，则：

$$
\|o-\widetilde o\|_2
\le
\eta\,\operatorname{diam}(V).
$$

因此，集合 recall 不是最关键指标；保留高 attention-mass token 比恢复所有
边界 token 更重要。128K 实验中最终 top-1 agreement 为 95.90%，平均
$KL(\text{Full}\|\text{Sparse})=0.01008$。

## 7. 层级尾部编码探索

### 7.1 固定位宽消融

跨 Qwen3-4B、Llama-3.1-8B、Qwen2.5-7B 的 sports/medicine trace：

| 分配 | 索引 / Full KV | top-1% recall | Oracle mass recall | Pearson |
|---|---:|---:|---:|---:|
| 固定 4-4-4 | 5.859% | 69.13% | 94.11% | 0.9540 |
| 固定 8-4-0 | 5.469% | 67.01% | 95.69% | 0.9503 |
| 自动 qMSE，budget=15 | 约 5.8% | **70.48%** | **96.36%** | **0.9616** |
| 固定 8-4-4 | 7.422% | 74.62% | 97.44% | 0.9710 |

用户提出的“重要频带高 bit、尾部低 bit”方向成立；但固定 8/4/4 增加约
29% 索引成本。该早期表没有包含后来发现的强固定端点 `4-4-2-1`。

在 Qwen3 sports/medicine 的严格同 240-bit、同 trace 对照中：

| 分配 | top-1% recall | Oracle mass recall | Pearson |
|---|---:|---:|---:|
| empirical auto qMSE | 70.37% | 96.40% | 0.9642 |
| regularized-diagonal auto qMSE | **70.77%** | **96.71%** | 0.9653 |
| fixed `4-4-2-1` | 70.50% | 96.59% | **0.9654** |

三者差异很小。regularized auto 在 proxy recall 上略高，但 fixed
`4-4-2-1` 的 Pearson 相当，并且 LongBench 开发切分略高于 empirical
auto。当前证据只支持“谱带不均匀分配优于均匀低比特”，还不支持“逐 prompt
自动分配必然优于固定分配”。这两者必须保留为正式主消融。

### 7.2 Key 中心化

对所有 Key 减去同一均值 $\mu$：

$$
q^\top(k_i-\mu)=q^\top k_i-q^\top\mu.
$$

$q^\top\mu$ 对全部 token 相同，因此排序和 softmax 精确不变。中心化在全部
trace 上改善了代理指标，但 128K 独立 PPL 保持率由 99.80% 降至
99.67%。该结果说明代理改进不保证最终 NLL 改进，中心化暂不进入主方法。

### 7.3 共享幅值 1-bit 尾部

测试方案：

- 前 32 维：INT4 + INT4；
- 第 32--96 维：每维 1-bit sign；
- 4 个尾部 band 共享一个 FP16 token envelope；
- 最后 32 维丢弃；
- 总计 240 bit。

其共享 envelope 为：

$$
e_i=
\sqrt{\frac1{64}\sum_{j=32}^{95}
\left(\frac{k'_{ij}}{r_j}\right)^2}.
$$

每个坐标幅值通过加权最小二乘求得：

$$
a_j=
\frac{\sum_ie_i|k'_{ij}|}{\sum_ie_i^2}.
$$

近似：

$$
\widehat k'_{ij}
=\operatorname{sign}(k'_{ij})e_ia_j.
$$

同预算 proxy top-1% recall 由约 73.14% 提升到 76.02%，但独立 128K
PPL 保持率只有 **99.2446%**，低于主方法 99.8010%。因此它只保留为
系统速度和“proxy/endpoint 不一致”的消融，不进入主方法。

### 7.4 无偏 JL 尾部草图：负结果

该方案把前 32 个高能坐标保持为两组 INT4，把后 96 维残差
$z$ 压缩为低维 Rademacher 草图：

$$
R_{jl}\in\{-1,+1\}/\sqrt m,
\qquad
\widehat s_{\rm tail}=(q_{\rm tail}^\top R)
(z^\top R)^\top.
$$

对随机矩阵的期望：

$$
E_R[\widehat s_{\rm tail}]
=q_{\rm tail}^\top z.
$$

其方差为：

$$
\operatorname{Var}(\widehat s_{\rm tail})
=\frac1m
\left(
\|q\|_2^2\|z\|_2^2
+(q^\top z)^2
-2\sum_jq_j^2z_j^2
\right)
\le\frac{2}{m}\|q\|_2^2\|z\|_2^2.
$$

固定总 tail 成本 80 bit 的三个配置为：

| JL 维数 | Key 位宽 | Tail 成本 | 总索引 |
|---:|---:|---:|---:|
| 8 | INT8 | 64 + 16 scale | 240 bit |
| 16 | INT4 | 64 + 16 scale | 240 bit |
| 32 | INT2 | 64 + 16 scale | 240 bit |

还测试 208-bit 的 `8×INT4` 和 `16×INT2`。每个 layer/head 同时比较固定
随机种子与只使用 prompt 校准 Query 选择 8 个种子中 qMSE 最小者；真正指标
在后续 Query 上评估，不能把校准 qMSE 当测试结果。

该方案比共享幅值符号尾部更有理论依据：未量化时 tail score 无偏，而共享
幅值重构一般有偏。但 held-out 结果表明，无偏不等于低方差：

| 方法 | 总 bit | top-1% recall | Oracle mass recall | Pearson |
|---|---:|---:|---:|---:|
| JL8 × INT8，select-8 | 240 | 31.69% | 76.35% | 0.7607 |
| JL16 × INT4，select-8 | 240 | 41.97% | 84.76% | 0.8408 |
| JL32 × INT2，select-8 | 240 | **47.94%** | **88.04%** | **0.8789** |
| 共享幅值 1-bit tail | 240 | 72.47% | 95.67% | 0.9682 |
| 标准自动 qMSE | 237.03 平均 | **71.82%** | **96.76%** | **0.9686** |

选择 8 个随机种子只能带来约 0.1--0.5 个 recall 百分点，无法弥补
$m\le32$ 的随机投影方差。该方向不做 CUDA 内核和端到端 PPL。它同时说明：
评价低 bit tail 不能只看 estimator 是否无偏，还必须控制 top-k 边界附近的
方差和最差 trace attention-mass recall。

### 7.5 Softmax-Fisher 位宽分配：负结果

均匀 qMSE 将所有 sampled token 的 score error 等权处理。为直接近似
softmax 输出的局部 KL，测试了 Fisher 成本。对校准 attention $p$ 和某一
频带的 score error $e$：

$$
D_{\rm Fisher}
=e^\top(\operatorname{diag}(p)-pp^\top)e
=\sum_ip_ie_i^2-\left(\sum_ip_ie_i\right)^2.
$$

它是：

$$
KL\!\left(p(s)\|p(s+te)\right)
=\frac{t^2}{2}D_{\rm Fisher}+O(t^3)
$$

中的二阶项，并精确忽略对全部 token 相同的 logit 偏移。实验仅替换
rate allocator 的成本函数，坐标、240-bit 预算、量化格式和 held-out Query
全部保持相同。

| 分配目标 | 平均 bit | top-1% recall | Oracle mass recall | Pearson | RMSE |
|---|---:|---:|---:|---:|---:|
| 均匀 score qMSE | 237.03 | **71.82%** | **96.76%** | **0.9686** | **0.7744** |
| Softmax-Fisher | 236.69 | 69.83% | 96.47% | 0.9622 | 0.8429 |

Fisher 分配显著偏向 `8-1-1-1` 和 `8-4-0`，在短 prompt 校准 Query 上保护
当前高概率 token，却降低了未来 Query 的总体排序泛化。因此不进入主方法。
该负结果也限定了理论：局部 KL 几何适合描述给定 Query 的小扰动，但不自动
等价于跨 Query 的 retrieval index 目标。

## 8. 速度分解

### 8.1 Attention 子系统

测试范围：一个 Qwen-like attention layer，32 Query heads、8 KV heads、
head dim 128。包含 packed scan、sampled threshold、compaction、候选精确
QK、当前 token、softmax 与 V 聚合，也包含 Query 量化；不含建索引和
非-attention 模型计算。

| 历史长度 | 标准 qMSE 索引 | 共享尾部索引 |
|---:|---:|---:|
| 2K | 0.752x | 0.750x |
| 4K | 1.190x | 1.181x |
| 8K | 1.701x | 1.817x |
| 16K | 2.292x | 2.501x |
| 32K | 4.067x | 4.354x |
| 64K | 5.599x | 6.408x |
| 128K | **7.239x** | **9.075x** |

共享尾部 kernel 更规则，所以更快；但其 PPL 较差，不能只凭系统速度替换
主方法。

### 8.2 安全渐进式频带读取

完整 `4-4-2-1` 索引每个 token/head 需要读取 15 个 rate unit。新做的
两阶段数值方案先对所有 token 只读取第一个 4-bit 主频带，再用 256 个均匀
样本估计未读残差的 conformal 半径，只对可能跨越当前 top-k 阈值的 token
读取其余 `4-2-1` 频带。

在 sports 与 medicine 的 32K trace 上，取开发阶段固定的
`base rate=5, alpha=0.05`：

| 目标候选率 | full-proxy top-k 包含率 | full-proxy mass 保持率 | 实际索引读比例 | 折合 Full KV |
|---:|---:|---:|---:|---:|
| 1% | 99.9972% | 99.9946% | 59.23% | 3.461% |
| 4% | 99.9988% | 99.9990% | 67.91% | 3.969% |

这些数字证明两阶段读取几乎精确复现**完整 packed proxy**，不是证明它等价于
exact QK oracle。它在 1% 候选率下理论上可少读约 40.8% 的索引字节。

但 3090 CUDA 实测否定了当前执行形式：

| 长度与候选设置 | Full scan+top-k | 随机 gather 两阶段 | 顺序 masked 两阶段 |
|---|---:|---:|---:|
| 32K，51% refinement | 0.156 ms | 0.551 ms，0.283x | 0.578 ms，0.270x |
| 64K，38% refinement | 0.244 ms | 0.744 ms，0.328x | 0.856 ms，0.285x |
| 128K，38% refinement | 0.394 ms | 1.274 ms，0.309x | 1.465 ms，0.269x |

单独主频带扫描有 `1.9x--2.8x` 加速，128K 完整 packed score 扫描本身却只有
0.184 ms。候选生成、第二次 top-k、mask/compaction 和不规则尾频带读取超过
了省下的带宽。由于 0.184 ms 相对整模型约百毫秒的 token 延迟可以忽略，
继续优化该子路径不会带来可见端到端收益；它保留为负结果，不进入主方法。

基于 token norm 的严格证书也完成了。它能做到 full-proxy top-k 最小包含率
100%，但最佳配置在 1%/4% 目标下仍需读取 85.37%/90.67% 的完整索引，明显
差于 conformal trace 的 59.23%/67.91%，且还要保存 norm metadata。因此
严格 norm 证书同样不进入主系统。

### 8.3 整模型长度交叉点

每个长度只使用一个 64-token mixed_a 窗口，因此 PPL 仅为 smoke test；
速度可用于定位交叉点。

| 长度 | Attention 子系统 | 整模型稳态 | 64-token 实测，含建索引 | break-even |
|---:|---:|---:|---:|---:|
| 4K | 1.190x | 1.070x | 0.514x | 971 token |
| 8K | 1.701x | 0.820x | 0.407x | 无 |
| 16K | 2.292x | 1.033x | 0.510x | 1940 token |
| 24K | 未测 | 1.381x | 0.666x | 177 token |
| 32K | 4.067x | 1.720x | 0.851x | 89 token |
| 48K | 未测 | 2.444x | 1.205x | 45 token |
| 64K | 5.599x | 3.093x | 1.547x | 30 token |
| 128K | 7.239x | **4.407x** | **3.707x，256 token** | **14 token** |

这说明当前系统的主要不足仍是短上下文：

- 8K attention kernel 已有 1.70x，但完整 HF 模型仍只有 0.82x；
- 固定建索引成本约 3.3 秒，在短生成场景无法摊销；
- 24K 以上稳态收益开始稳定，48K 以上短生成也能兑现端到端加速。

上表是早期单窗口长度扫描，用于定位趋势。新完成的 32K 四主题物理缓存配对
使用 fused directory、CPU exact-K/V offload 和 64 个同步在线 token，得到
QK-balanced full-topk `3.509x` 稳态整模型加速；但其平均额外建索引成本也
增加到 43.62 秒，所以有限生成长度的 break-even 约为 120 token。论文中
必须分别报告稳态、固定成本和给定生成长度的总延迟，不能只选其中一个。

## 9. RaBitQ 同 trace 对照

在 8 个跨模型/领域 trace、59,200 个严格配对的 held-out
layer/head/query 样本上，双方都选择 1% token：

| 方法 | top-1% recall | 选中 attention mass | Score Pearson |
|---|---:|---:|---:|
| RaBitQ，INT4 Query 公式模拟 | 53.88% | 71.32% | 0.9087 |
| 当前 240-bit qMSE 谱索引 | **71.82%** | **76.79%** | **0.9686** |
| 配对差值 | **+17.94 pp** | **+5.47 pp** | **+0.0600** |

块 bootstrap 的差值 95% CI：

- recall：`[+17.49, +18.40]` 个百分点；
- selected attention mass：`[+5.09, +5.87]` 个百分点；
- Pearson：`[+0.0572, +0.0623]`。

RaBitQ 论文口径只计 binary code 和 alpha，为 3.516% Full KV；若按其实际
score 分解额外计入 key norm 和每个 GQA group 的 $c_q^\top k$ 状态，本批
trace 加权平均为 5.779%，与本方法 5.80% 接近。

该结果仍是**公式级同 trace 模拟**，不是官方 RaBitQ CUDA 代码的完整
LongBench/PPL/速度复现。论文中只能作为机制对照，不能替代官方 baseline。

## 10. 与近期工作的边界

不能声称：

- 首个 token-level KV retrieval；
- 首个低 bit Key 索引；
- 首个 1-bit Key 或随机旋转二值检索；
- 首个 mixed-precision KV 方法；
- 首个 PCA/SVD 或低秩 QK 方法；
- 首个 rate-distortion 位宽分配。

最相关工作：

- FIER：1-bit Key 的 token-level retrieval，约 11% cache budget，
  1.2x--1.5x decode speed；
- RaBitQCache：随机旋转 binary Key、INT4 Query、无偏 score estimator；
- Block-GTQ：RoPE 二维 block 的 mixed-bit Key 量化；
- RateQuant：跨 head/quantizer 的 rate-distortion 位宽分配；
- Quantization Dominates Rank Reduction：使用 softmax Fisher metric
  比较保留全维低比特与低秩截断；
- Loki、LRQK：低维 QK/Key 空间中的候选检索；
- QPruningKV：token 数量与量化精度的联合权衡。

当前可辩护的组合贡献是：

1. **Query-conditioned biorthogonal score coordinates**：满维严格保持
   $q^\top k$，截断直接优化二阶模型下的 score MSE。
2. **Within-head physical rate allocation**：在一个 head 内对谱带做
   0/1/2/4/8-bit 分配，并把 FP16 scale metadata 纳入预算。
3. **Direct candidate consumption**：没有 learned router、任务规则、
   Full fallback 或 exact rerank。
4. **端到端数值与系统闭环**：约 1.05% attention token、5.86% 索引，
   128K 逻辑路径 PPL 99.94%、4.31x 稳态；物理 offload 路径质量
   99.88%--99.96%、GPU KV 约 9.3%、在线 1.85x--2.24x。新增的 group16
   临时 KV + 直接 K 转码在单个 128K 物理窗口上达到 99.06% 质量、2.23x
   在线和 1.302x 的 256-token 总加速；该单窗口结果不能替代待完成的四窗口
   物理复测。

参考：

- FIER: https://aclanthology.org/2025.findings-emnlp.515/
- RaBitQCache: https://arxiv.org/abs/2606.31519
- Block-GTQ: https://arxiv.org/abs/2606.24033
- RateQuant: https://arxiv.org/abs/2605.06675
- Quantization Dominates Rank Reduction: https://arxiv.org/abs/2604.11501
- QPruningKV: https://aclanthology.org/2025.findings-emnlp.429/

## 11. 距离 ICLR 8/10 还缺什么

当前方法创新性已经不再只是“PCA48 + INT4”，但不能把局部好结果直接等同于
8/10。优先级如下。

### P0：投稿前必须闭环

1. Llama-3.1-8B 的完整 16-task query-preserving LongBench 已完成；
   `official_middle` 七路正式实验和 Qwen2.5 第二模型实验已排队。
2. RULER 8K--128K，覆盖多种 NIAH、变量追踪和聚合任务。
3. 在同模型、同硬件、同上下文、同候选率下运行 FIER、RaBitQCache、
   Loki/LRQK 类强 baseline 的官方实现。
4. 32K/128K CPU exact-K/V offload 物理路径均已完成；group16 临时 KV +
   直接 K 转码已在一个 128K 窗口把固定阶段降到 Full 以下、256-token 总加速
   提到 1.302x。还需完成四窗口复测，并明确报告 pinned CPU、转换后 GPU KV
   和 prefill 峰值三种占用。
5. 对完整 benchmark 报告质量、索引成本、实际候选数、attention
   subsystem、整模型稳态、有限生成长度和峰值显存。

### P1：显著提高评分

1. 三个模型和至少两种硬件，验证 qMSE 分配不是 Qwen3/RTX3090 特例。
2. 对 shrinkage、240-bit rate、采样数、候选预算做独立验证集上的敏感性。
3. 报告 paired bootstrap，不把 layer/head 当作相互独立样本。
4. 解决 8K/16K 的 HF 集成开销，或明确限定目标域为 24K 以上。
5. 开源复现脚本、固定 seed、原始逐样本结果和 CUDA correctness tests。

## 12. 当前冻结判断

冻结的方法骨架与配置为：

- `qk_metric_query_shrinkage=0.75`
- `total rate units=15`
- `candidate_overfetch=1.0`
- `B(N)=min(N,1280,max(256,ceil(0.06N)))`
- `no rerank`
- `no fallback`
- `no router`
- prefill 临时 K/V：`INT4, group_size=16`
- 物理转换：`transient_quantized_key` 直接转码并复用 pinned host K/V

当前稳健逻辑参考是：

`pca_hierarchical_autoqmsetotal15z_qkmetric_packed_direct`

当前 32K/128K 在线速度参考仍是 QK-balanced packed index + `full-topk`。
单窗口固定成本最优参考已更新为 group16 临时 KV + 直接 K 转码 + fixed
`4-4-2-1` + `sampled-compact`；下一项系统实验是把相同 prefill 路径与
`full-topk`/auto qMSE 组合，而不是重新调候选率。以下两个逻辑选择中只剩
第一个尚未冻结：

1. auto qMSE 与 fixed `4-4-2-1`；
2. **plain scale 已冻结**；raw Q-aware 与 OAS Q-aware 都是消融。

`sampled-compact` 在 32K 和 128K 都没有快过 `full-topk`，因此不再假定存在
当前实现下的长度交叉点。渐进式频带读取的两种 CUDA 实现也均慢于
full-topk，因此停止该方向。

raw Q-aware scale
`pca_hierarchical_autoqmsetotal15z_qkmetric_qscale_packed_direct`
保留为 128K 专用消融，不再作为通用默认。只有下面两种证据之一出现时才替换
稳健参考：

1. 新方案在跨模型 held-out proxy 上稳定优于主方法，并在独立 128K PPL
   上不低于 99.8%；
2. 在质量相当时，真实整模型速度或物理 KV 显存有显著改善。

共享尾部、JL 尾部、raw Q-aware 和 OAS Q-aware 均没有通过第一条，因此都
不替换主方法。

## 13. 代码与结果

核心代码：

- `src/run_head_top2_targeted_ppl_20260714.py`
- `src/variablebit_spectral_cuda_20260727.py`
- `src/run_sample_calibrated_longbench_20260717.py`
- `src/hierarchical_pca_cache_20260715.py`
- `src/run_hierarchical_physical_cache_ppl_20260715.py`
- `src/analyze_qkbalanced_longbench_paired_20260727.py`
- `src/analyze_qkbalanced_factorial_20260727.py`
- `src/analyze_qk_matched_rate_all_dims_20260727.py`
- `src/analyze_qk_norm_certified_refinement_20260727.py`
- `src/analyze_qkbalanced_additivity_closure_20260727.py`
- `src/progressive_variablebit_cuda_20260727.py`
- `src/benchmark_progressive_variablebit_pipeline_20260727.py`
- `src/offloaded_prefill_cache_20260716.py`
- `src/analyze_groupwise_prefill_quantization_20260727.py`

验证：

- `tests/test_countcap_fullprompt.py`
- `tests/test_packed_qmse_short_history_20260727.py`
- `tests/test_qkbalanced_additivity_closure_20260727.py`
- `tests/test_qkbalanced_factorial_20260727.py`
- `tests/test_qk_matched_rate_all_dims_20260727.py`
- 2026-07-27 当前相关回归组合共 117 项通过；
- CUDA shared-tail 和融合 Query 准备的最大 score 误差为
  $2.33\times10^{-6}$。

主要结果：

- `results/20260727_qkmetric_fusedquery_128k_holdout`
- `results/20260727_fusedquery_both_operator_benchmark.json`
- `results/20260727_qkmetric_fusedquery_length_sweep`
- `results/20260727_qkmetric_fusedquery_speed_summary`
- `results/20260727_rabitq_top1_crossmodel`
- `results/20260727_rabitq_top1_crossmodel_comparison`
- `results/20260727_sharedtail240_centered_128k_holdout`
- `results/20260727_jl_tail_crossmodel`
- `results/20260727_qkmetric_qscale_128k_holdout`
- `results/20260727_qkbalanced_longbench_official75k_full_8gpu`
- `results/20260727_qkbalanced_allocation_scale_factorial_m20_5gpu`
- `results/20260727_qk_variable_physical_32k_paired_8gpu`
- `results/20260727_qkbalanced_qscale_oas_dev_m20_5gpu`
- `results/20260727_qk_norm_certified_refinement_32k`
- `results/20260727_qkbalanced_additivity_closure_32k`
- `results/20260727_progressive_variablebit_coalesced_3090`
- `results/20260727_groupwise_int4_prefill_32k_paired_4gpu`
- `results/20260727_direct_quantized_key_conversion_3gpu`
- `results/20260727_direct_quantized_key_32k_paired_4gpu`
- `results/20260727_groupwise_prefill_mechanism_hybrid_gpu1`
