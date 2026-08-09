# CountCap 的谱稳定性：数学推导与可证边界

更新时间：2026-07-26

## 1. 要证明的不是“PCA 后答案必然不变”

当前方法使用低维、量化的 Key 索引选择候选 token，然后在候选集合内使用原始 FP16 Q/K/V 计算 attention。合理的理论目标是建立下面的条件链：

$$
\text{K 的低秩谱结构}
\Longrightarrow
\text{QK 代理分数误差受控}
\Longrightarrow
\text{候选集合遗漏的 attention mass 较小}
\Longrightarrow
\text{attention 输出扰动受控}
\Longrightarrow
\text{logit、NLL 和任务输出稳定}.
$$

其中前三个箭头可以得到严格公式；最后一个箭头依赖模型在当前输入附近的 residual 和 logit margin，必须同时用实验验证。任何有限维截断都无法对任意 query、任意输入保证答案不变。

---

## 2. Uncentered PCA 与截断 SVD 是同一个子空间

考虑某层、某个 KV head 的 post-RoPE 历史 Key：

$$
K\in\mathbb R^{N\times d},
$$

其中 $N$ 是历史 token 数，当前模型中 $d=128$。对 $K$ 做奇异值分解：

$$
K=U\Sigma V^\mathsf T,\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_d),
\quad \sigma_1\ge\cdots\ge\sigma_d\ge0.
$$

未中心化 PCA 使用二阶矩：

$$
C_K=\frac{1}{N}K^\mathsf TK
=V\frac{\Sigma^2}{N}V^\mathsf T.
$$

因此，未中心化 PCA 的特征向量就是 $K$ 的右奇异向量。令

$$
V_r=[v_1,\ldots,v_r],\qquad P_r=V_rV_r^\mathsf T,
$$

则 rank-$r$ 主子空间近似为

$$
K_c=KP_r=U_r\Sigma_rV_r^\mathsf T,
$$

残差为

$$
K_t=K(I-P_r)=\sum_{j>r}\sigma_j u_jv_j^\mathsf T.
$$

所以：

> 如果 PCA 使用完整 $K^\mathsf TK$ 且不减均值，那么 PCA48 与 full SVD48 数学上完全等价。此前 SVD 和 PCA 实验效果接近，不是偶然现象。

冻结生产实现使用 2048-token chunked prefill。首个 chunk 结束后的增量索引回调
从该首段每隔 32 个 token 采样一个 Key 来估计基底；基底建立后不再旋转，只将
后续 Key 增量投影进去。因此生产版更准确地说是 **first-2K sampled uncentered
prefix-PCA48**，而不是对完整历史的精确或 sampled full-SVD48。

本节关于 full $K$ 的 SVD/PCA 等价仍然成立，但它描述的是理想 full-history
子空间。第 4.4 节进一步把真实 prefix basis 与该理想子空间之间的失配单独列出。

---

## 3. “奇异”的是谱集中，不是第 48 个向量

本文用谱能量分布定义有效秩：

$$
p_j=\frac{\sigma_j^2}{\sum_\ell\sigma_\ell^2},
\qquad
r_{\mathrm{eff}}
=
\exp\left(-\sum_jp_j\log p_j\right).
$$

因此该量同时考虑全部奇异值，而不是使用某个任意阈值统计非零秩。

32K 真实 Q/K/V 探针覆盖 80 个 layer-KV-head，得到：

| 谱指标 | 结果 |
|---|---:|
| Key 维度 | 128 |
| 平均有效秩 | 24.67 |
| 前 48 维保留 Key 谱能量 | 86.84% |
| 前 48 维保留能量 p10 | 81.52% |
| $\sigma_1/\sigma_{48}$ | 10.73 |
| $\lambda_{48}/\lambda_{49}$ | 1.021 |
| sampled PCA48 与 full SVD48 的平均子空间重合度 | 91.50% |

这里真正重要的是：

1. $d=128$，但平均有效秩只有 24.67，说明谱能量高度集中。
2. 前 48 维已经覆盖大部分 Key 能量。
3. $\lambda_{48}/\lambda_{49}$ 接近 1，说明第 48 与第 49 个方向之间没有明显 eigengap。
4. 因此不能声称“第 48 个奇异向量具有特殊语义”。边界附近的单个奇异向量可以旋转，较稳定的是整个 dominant subspace 及其完成检索的能力。

这也解释了为什么采样 PCA 与 full SVD 的结果接近：即使边界附近的单个向量不同，只要两个 rank-48 子空间整体接近，投影后的 QK 排序仍可相似。

---

## 4. PCA 截断如何影响 QK 分数

令当前 query 为 $q\in\mathbb R^d$，并分解为

$$
q_c=P_rq,\qquad q_t=(I-P_r)q.
$$

精确分数向量为

$$
s=\frac{Kq}{\sqrt d}.
$$

由于主子空间与尾子空间正交：

$$
s
=
\frac{K_cq_c}{\sqrt d}
+
\frac{K_tq_t}{\sqrt d}.
$$

PCA 分数为

$$
s^{\mathrm{PCA}}=\frac{K_cq_c}{\sqrt d},
$$

截断误差精确等于

$$
\delta^{\mathrm{PCA}}
=s-s^{\mathrm{PCA}}
=\frac{K_tq_t}{\sqrt d}.
$$

利用 SVD 可得：

$$
\left\|\delta^{\mathrm{PCA}}\right\|_2^2
=
\frac{1}{d}
\sum_{j>r}\sigma_j^2(v_j^\mathsf Tq)^2.
$$

这个式子说明，仅报告“PCA48 保留 86.84% Key 能量”还不够。真实误差同时由两个因素决定：

- 尾部奇异值 $\sigma_j$；
- query 在对应尾部右奇异向量上的投影 $v_j^\mathsf Tq$。

实验中 query 平均只有 65.16% 的自身能量落在前 48 维，但 PCA 尾部 score energy 只占精确 score energy 的 10.63%。原因正是 query 尾分量还要乘上较小的尾部奇异值。

### 4.1 Softmax 有效的 QK 矩阵必须先按行中心化

原始 QK 矩阵可能包含一个很大的、但对 attention 完全无效的均值模态。令

$$
H=I_N-\frac{1}{N}\mathbf 1\mathbf 1^\mathsf T,
\qquad
\widetilde K=HK.
$$

对分数矩阵

$$
S=\frac{QK^\mathsf T}{\sqrt d}
$$

按历史 token 维度做行中心化，得到

$$
S^\circ
=SH
=\frac{Q\widetilde K^\mathsf T}{\sqrt d}.
$$

因为 $SH$ 只是从每个 query 的全部 token 分数中减去该行均值，

$$
\boxed{
\operatorname{softmax}(S)
=
\operatorname{softmax}(S^\circ)
}.
$$

因此，若 Key 均值 $\mu=N^{-1}\sum_i k_i$ 很大，原始 $S$ 中会出现
$Q\mu\mathbf 1^\mathsf T/\sqrt d$ 这一至多 rank-1 的大能量分量。它会让原始
QK 的有效秩看起来接近 1，却不会改变任何 attention 概率、top-$k$ 排序或输出。
论文中的核心谱证据必须使用 $S^\circ$，不能把这个 softmax 无效模态算作压缩收益。

令任意 rank-$r$ 投影为 $P$，代理分数为

$$
\widehat S=\frac{QPK^\mathsf T}{\sqrt d}.
$$

真正影响 softmax 的误差是

$$
\boxed{
E^\circ
=(S-\widehat S)H
=
\frac{Q(I-P)\widetilde K^\mathsf T}{\sqrt d}
}.
$$

又因为 $H$ 是正交投影，

$$
\|E^\circ\|_F
=
\|(S-\widehat S)H\|_F
\le
\|S-\widehat S\|_F.
$$

所以未中心化 SVD 尾部公式仍然给出一个正确但可能偏松的上界；真正对应候选排序
与 softmax 的紧致量，应直接在 $\widetilde K$ 或 $S^\circ$ 上测量。

这也给出比“common + tail”更精确的三项分解。令

$$
\mu=\frac{1}{N}K^\mathsf T\mathbf1,
\qquad
K=\mathbf1\mu^\mathsf T+\widetilde K,
$$

并对任意检索投影 $P$ 写成

$$
\widetilde K
=
\underbrace{\widetilde KP}_{\text{保留的中心化主子空间}}
+
\underbrace{\widetilde K(I-P)}_{\text{中心化谱尾部}}.
$$

则精确分数可以写为

$$
\boxed{
S
=
\underbrace{\frac{Q\mu\mathbf1^\mathsf T}{\sqrt d}}_
{\text{softmax 精确忽略的 token 公共项}}
+
\underbrace{\frac{QP\widetilde K^\mathsf T}{\sqrt d}}_
{\text{低维索引保留项}}
+
\underbrace{\frac{Q(I-P)\widetilde K^\mathsf T}{\sqrt d}}_
{\text{真正影响排序的谱尾误差}}
}.
$$

只有第一项可以无条件忽略；第三项并非“没有语义”，只能在中心化谱尾能量小、
top-$k$ 边界不承载大量 attention mass、且下游 logit margin 足够时近似忽略。
INT4/INT8 则是在第二项坐标中的额外数值扰动，不应与第三项混为同一种语义尾部。

令 $S=U\Sigma V^\mathsf T$，第一左右奇异向量为 $u_1,v_1$。第一右奇异向量
与 token 常数方向的平方对齐度可在不显式构造长矩阵 $V$ 的情况下计算：

$$
\rho_1^2
=
\left|
v_1^\mathsf T\frac{\mathbf1}{\sqrt N}
\right|^2
=
\frac{|u_1^\mathsf TS\mathbf1|^2}{N\sigma_1^2}.
$$

被行中心化精确删除的无效能量比例为

$$
\alpha_{\mathrm{mean}}
=
\frac{\|S(I-H)\|_F^2}{\|S\|_F^2}
=
\frac{\|S\mathbf1\|_2^2}{N\|S\|_F^2}.
$$

32K、Llama-3.1-8B/Qwen3-4B/Qwen2.5-7B 的 200 个
topic-layer-KV-head 实测结果为：

| 模型 | 原始 rank-1 能量 | 行中心化删除的无效能量 | 第一右奇异向量与常数方向对齐 | 中心化 rank-1 能量 |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 97.19% | 88.83% | 91.13% | 70.89% |
| Qwen3-4B | 82.44% | 62.70% | 70.60% | 67.18% |
| Qwen2.5-7B | 95.63% | 90.51% | 93.55% | 63.58% |

| 模型 | 原始 QK 有效秩 | 中心化 QK 有效秩 | 中心化 QK 最优 rank-48 能量 | 未中心化 Key-PCA48 保留的中心化能量 | 生产 first-2K basis 保留的中心化能量 | 生产 basis 中心化 score cosine |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 1.22 | 4.72 | 99.09% | 91.10% | 62.39% | 0.7953 |
| Qwen3-4B | 2.79 | 5.43 | 98.52% | 91.45% | 59.07% | 0.7609 |
| Qwen2.5-7B | 1.41 | 6.29 | 98.80% | 91.67% | 60.37% | 0.7864 |

中心化 QK 自身的最优累计能量在 rank-16 时已达到
Llama/Qwen3/Qwen2.5 的 94.79%/93.50%/94.00%，rank-32 时达到
98.01%/97.08%/97.61%。这证明“存在一个很低维的好分数
子空间”，但该最优子空间使用了完整 $Q$ 与 $K$，属于解释上界，不是冻结方法
可以直接在线获得的 basis。

分布尾部进一步区分了“QK 低秩”和“生产 basis 准确”这两个命题。中心化 QK
最优 rank-48 的 p10 为 98.30%/97.74%/97.88%，最低值为
97.69%/95.11%/97.48%，低秩结构跨 head 较稳定；但生产 first-2K
fidelity 的 p10 只有 39.58%/31.22%/42.72%，最低值为
9.88%/23.88%/7.30%。因此不能用平均 fidelity 掩盖困难 head，最终证据必须落到
mass-weighted recall、attention 输出和 logit/NLL。

这组结果支持三个不同强度的结论：

1. 原始第一奇异模态确实高度对齐 token 常数方向；Llama/Qwen3/Qwen2.5
   平均分别有 88.83%/62.70%/90.51% 的原始 QK 能量被行中心化删除。Qwen3 的该比例 p10 只有
   22.40%，所以这不是每个 head 都成立的统一现象。
2. 去掉无效均值模态后，QK 仍然具有很强的低秩结构，因而低维检索有真实依据。
3. QK 自身可被 rank-48 很好地表示，不等于只看 Key 的 PCA48 已经达到最优；
   Key-only 子空间与 query-aware 的最优 QK 子空间仍有约 7--8 个百分点能量差距。
4. 生产实现的主要近似误差不是 INT4，而是仅用首 2K、64 个 Key 样本冻结 basis
   带来的跨位置分布失配。最终质量仍好，依赖的是高 attention-mass 保留率和
   下游 logit 稳定，而不是生产代理重构了全部中心化 QK 能量。

额外诊断中，直接对完整历史的中心化 Key 做 PCA48，中心化 QK 保能率为
Llama 88.59%、Qwen 91.50%，并未稳定优于当前未中心化 Key-PCA。原因是
Key-PCA 无论是否中心化，都没有使用 query 协方差 $Q^\mathsf TQ$；中心化本身
不能保证得到 QK 加权意义下的最优子空间。

### 4.2 从单个 query 推广到完整 QK 分数矩阵

把同一 KV head 的 $M$ 个 decode query 堆叠为

$$
Q\in\mathbb R^{M\times d},
$$

则完整分数矩阵为

$$
S=\frac{QK^\mathsf T}{\sqrt d}
=
\frac{QV\Sigma U^\mathsf T}{\sqrt d}.
$$

Key-PCA rank-$r$ 近似产生

$$
S_r
=
\frac{QV_r\Sigma_rU_r^\mathsf T}{\sqrt d}.
$$

其 Frobenius 误差满足精确等式

$$
\boxed{
\|S-S_r\|_F^2
=
\frac{1}{d}
\sum_{j>r}
\sigma_j^2\|Qv_j\|_2^2
}.
$$

同时有谱范数和 Frobenius 范数上界：

$$
\|S-S_r\|_2
\le
\frac{\sigma_{r+1}(K)\|Q\|_2}{\sqrt d},
$$

$$
\|S-S_r\|_F^2
\le
\frac{\sigma_{r+1}^2(K)\|Q\|_F^2}{d}.
$$

这说明不能只看 $K$ 的奇异值，也不能只看 $Q$ 的能量。真正控制 QK 矩阵尾部
的是乘积

$$
\sigma_j^2\|Qv_j\|_2^2.
$$

如果 $Q^\mathsf TQ$ 与 $K^\mathsf TK$ 共享右奇异向量 $V$，记

$$
\mu_j=\|Qv_j\|_2^2,
$$

则 QK 矩阵的非零奇异值可写成

$$
\sigma_j(S)
=
\frac{\sigma_j(K)\sqrt{\mu_j}}{\sqrt d}.
$$

因此，实验中所说的“QK 奇异谱更集中”可能来自两层作用：

1. $K$ 自身的尾部奇异值较小；
2. 自然 decode query 在这些 Key 尾部方向上的总能量也较小。

一般情况下 $Q^\mathsf TQ$ 与 $K^\mathsf TK$ 不完全对易，QK 的奇异向量不会与
Key 奇异向量逐个相同。此时 QK 的非零奇异值平方等于下面矩阵的特征值：

$$
\frac{1}{d}
\Sigma V^\mathsf TQ^\mathsf TQV\Sigma.
$$

所以严谨实验应同时报告：

- $K$ 的有效秩和 PCA48 能量；
- $QK^\mathsf T$ 的有效秩；
- Key-PCA48 对 QK 矩阵能量的实际保留率；
- QK 自身最优 rank-48 的能量上界；
- $Q^\mathsf TQ$ 分别与 $K^\mathsf TK$、$\widetilde K^\mathsf T\widetilde K$
  的归一化 commutator。

若 Key-PCA48 接近 QK 的最优 rank-48 上界，说明只看 Key 建索引已经足够；若
二者差距大，则不能只用 Key 谱集中解释全部现象。当前实验在每个主题采集
64 个连续 decode 位置；由于两个模型每个 KV head 对应 4 个 GQA query head，
每个 QK 谱实际堆叠 $M=256$ 个 query、特征维度为 128，因此矩阵秩上限仍为
128，而不是由 64 个时间步人为限制为 64。双模型验证已经完成。

令

$$
\chi(A,B)=\frac{\|AB-BA\|_F}{\|A\|_F\|B\|_F}.
$$

原始/中心化 Key 协方差与 Query 协方差的 32K 实测交换子为：

| 模型 | 原始 Key $\chi$ | 中心化 Key $\chi$ |
|---|---:|---:|
| Llama-3.1-8B | 0.594 | 0.154 |
| Qwen3-4B | 0.151 | 0.124 |

Llama 的大幅下降说明 softmax 无效的 Key 均值方向也是原始奇异向量失配的重要
来源。中心化后两个模型仍不完全对易，因此 Key-PCA 与 query-aware QK-SVD
不会逐向量相同。较小交换子只说明两者具有“近似共享方向”的可能性；若没有
谱隙条件，它本身还不足以推出逐向量或 rank-$r$ 子空间接近。

这个条件可以写得更精确。令
$A^\circ=V\operatorname{diag}(a_1,\ldots,a_d)V^\mathsf T$，
$a_1\ge\cdots\ge a_d$，并在同一特征基中记
$B'=V^\mathsf TBV$。保留/丢弃子空间之间的 Query 协方差跨块为
$B'_{12}$。交换子在该跨块上的每个元素满足

$$
\left(V^\mathsf T[A^\circ,B]V\right)_{ij}
=(a_i-a_j)B'_{ij},
\qquad i\le r<j.
$$

若 rank-$r$ 边界存在分离

$$
\delta_r=\min_{i\le r<j}|a_i-a_j|>0,
$$

则有

$$
\boxed{
\|B'_{12}\|_F
\le
\frac{\|[A^\circ,B]\|_F}{\delta_r}
}.
$$

因此，“交换子小 + Key 谱边界有足够间隔”才控制 Query 在 Key-PCA
保留/丢弃子空间之间的混合。当前 rank-48 边界
$\lambda_{48}/\lambda_{49}=1.021$，谱隙很小，所以上式的最坏情况界会很松。
这解释了两个看似矛盾的现象：整体 dominant subspace 和累计 QK 能量可以稳定，
但第 48 个附近的单个奇异向量及其精确 top-$k$ 成员可能明显波动。

上面的等式是原始分数矩阵的精确恒等式。对 softmax 有效分数，只需把误差写为
$E^\circ=(S-S_r)H$；由 $\|E^\circ\|_F\le\|S-S_r\|_F$，原式仍是保守上界，
而紧致实验值使用第 4.1 节定义的中心化 QK。

### 4.3 Key-PCA 与 QK 最优低秩近似的精确差距

令

$$
A^\circ=\widetilde K^\mathsf T\widetilde K,
\qquad
B=Q^\mathsf TQ,
$$

以及

$$
G^\circ
=
\frac{1}{d}
(A^\circ)^{1/2}B(A^\circ)^{1/2}.
$$

$G^\circ$ 的特征值就是
$S^\circ=Q\widetilde K^\mathsf T/\sqrt d$ 的奇异值平方。由
Eckart--Young--Mirsky 与 Ky Fan 原理，中心化 QK 自身最优 rank-$r$ 近似
的平方残差与保留能量分别为

$$
\varepsilon_r^\star
=
\sum_{j>r}\lambda_j(G^\circ),
\qquad
E_r^\star
=
\sum_{j=1}^{r}\lambda_j(G^\circ).
$$

给定任意 rank-$r$ Key 子空间投影 $P_r$，其近似
$QP_r\widetilde K^\mathsf T/\sqrt d$ 的真实平方残差为

$$
\varepsilon_r^{K}
=
\frac{1}{d}
\left\|Q(I-P_r)\widetilde K^\mathsf T\right\|_F^2
=
\frac{1}{d}
\operatorname{tr}\left((I-P_r)A^\circ(I-P_r)B\right).
$$

因此存在一个可直接测量的、非负的最优性后悔值：

$$
\boxed{
\mathcal R_r
=
\varepsilon_r^{K}-\varepsilon_r^\star
\ge 0
}.
$$

归一化后，

$$
\overline{\mathcal R}_r
=
\frac{\varepsilon_r^{K}-\varepsilon_r^\star}{\|S^\circ\|_F^2}
=
\mathcal F_r^\star-\mathcal F_r(P_r),
$$

就是实验中的 `centered_qk_uncentered_key_pca_optimality_gap`。这个量比
“Key PCA 保留了多少 Key 能量”
更直接：它回答了只根据 Key 建立子空间，与可以看见全部 decode query 后再做
最优 QK-SVD 相比，额外增加了多少 score-matrix 残差。

这里必须使用残差定义。一般情况下，

$$
\left\|S^\circ\right\|_F^2
\ne
\frac{1}{d}\left\|QP_r\widetilde K^\mathsf T\right\|_F^2
+
\frac{1}{d}\left\|Q(I-P_r)\widetilde K^\mathsf T\right\|_F^2,
$$

因为两部分未必 Frobenius 正交。因此不能无条件把第一项的范数平方称为
“保留能量”。当 $P_r$ 是同一个 $A^\circ$ 的谱投影，或满足相应对易/正交
条件时，交叉项才为零，两种定义才一致。冻结实现使用未中心化、首段估计的
投影，实验必须按 $1-\varepsilon_r^K/\|S^\circ\|_F^2$ 报告 fidelity。

QK-SVD 上界为何通常不能由普通 Key 正交投影达到，也可以显式看出。先假设
$A^\circ$ 在其支撑空间上可逆，令 $W_r$ 是 $G^\circ$ 的前 $r$ 个正交
特征向量，定义

$$
T_r^\star
=
(A^\circ)^{1/2}
W_rW_r^\mathsf T
(A^\circ)^{-1/2}.
$$

则 $(T_r^\star)^2=T_r^\star$、$\operatorname{rank}(T_r^\star)=r$，并且

$$
\boxed{
\frac{QT_r^\star\widetilde K^\mathsf T}{\sqrt d}
=
(S^\circ)_r
},
$$

其中 $(S^\circ)_r$ 是中心化 QK 的 truncated rank-$r$ SVD。证明方法是注意
$S^{\circ\mathsf T}S^\circ$ 的前 $r$ 个右奇异向量可写成

$$
\widetilde K(A^\circ)^{-1/2}W_r,
$$

再把 $S^\circ V_rV_r^\mathsf T$ 展开。若 $A^\circ$ 奇异，使用
Moore--Penrose 伪逆并限制在其支撑空间即可。

一般情况下 $T_r^\star$ 不对称，因此是 Key 协方差白化坐标中的 query-aware
斜投影，而不是冻结方法允许的正交 Key-PCA 投影。只有当 $A^\circ$ 与 $B$
对易时，它才退化成共同特征方向上的正交选择。这个显式构造严格区分了三件事：

1. 同一个 Key 矩阵上的 PCA 与 right-SVD 完全等价；
2. Key-PCA 是受限的、query-agnostic 正交投影；
3. QK-SVD 是利用整批 decode Query 后才可得到的 query-aware 最优上界。

如果 $A^\circ$ 与 $B$ 对易，它们可以同时对角化。设对应特征值为
$a_j,b_j$，则中心化 QK
奇异值平方组成的集合为

$$
\left\{\frac{a_jb_j}{d}\right\}_{j=1}^{d}.
$$

中心化 QK 最优 rank-$r$ 选择最大的 $a_jb_j$。若使用 centered Key-PCA，
它选择最大的 $a_j$；冻结方法则使用未中心化 $K^\mathsf TK$ 的主方向。
两者接近不仅要求 Key 谱衰减，还要求均值方向不浪费过多维度，并且 query
协方差 $b_j$ 不把尾部方向重新排序到前面。
这正是为什么必须测量 QK 谱，而不能只凭 K 的谱作结论。

实验同时报告中心化 QK 在 rank 8、16、24、32、48、64 的累计能量、有效秩、
Key-PCA48 保能率和上述最优性后悔值。这里“QK
高度奇异”应严格表述为 **奇异值谱快速衰减、有效秩较低**；奇异向量本身没有
大小，且在谱间隙很小时并不唯一。

### 4.4 真实首段 basis 的额外失配项

令 $P_r$ 是完整历史 Key 的 rank-$r$ 对照投影，$\widehat P_r$ 是冻结实现根据
首 2048 token、stride 32 样本得到的投影。真实代理的未量化分数矩阵为

$$
\widehat S_r
=
\frac{Q\widehat P_rK^\mathsf T}{\sqrt d}.
$$

其 softmax 有效误差可以精确拆成：

$$
S^\circ-\widehat S_rH
=
\underbrace{\frac{Q(I-P_r)\widetilde K^\mathsf T}{\sqrt d}}_
{\text{完整历史的内在 rank-}r\text{ 截断误差}}
+
\underbrace{\frac{Q(P_r-\widehat P_r)\widetilde K^\mathsf T}{\sqrt d}}_
{\text{首段采样 basis 的分布失配}}.
$$

从而有

$$
\boxed{
\|S^\circ-\widehat S_rH\|_F
\le
\|S^\circ-S_rH\|_F
+
\frac{
\|Q\|_2
\|P_r-\widehat P_r\|_2
\|\widetilde K\|_F
}{\sqrt d}
}.
$$

第一项由 QK 奇异谱衰减控制，第二项则取决于首段是否代表后续历史。它说明
“完整 K 低秩”本身还不足以证明生产索引可靠；必须额外测量：

- 首段 basis 与 full-Key PCA48 的 principal-angle overlap；
- 首段 basis 对完整中心化 $QK^\mathsf T$ 的 relative MSE；
- 首段近似分数与完整中心化 QK 分数的 cosine；
- 首段 basis 与 sampled-full-history basis 的差距。

如果这些量跨主题、跨模型都稳定，才支持“请求首段足以估计后续检索子空间”
这一更强、也更贴近真实实现的经验命题。

### 4.5 RoPE 使首段 basis 稳定性成为独立问题

令 $\bar k_i$ 是位置 $i$ 的 pre-RoPE Key，$R_i$ 是对应的分块正交旋转，则
post-RoPE Key 为

$$
k_i=R_i\bar k_i,
\qquad
R_i^\mathsf TR_i=I.
$$

单个位置上的 RoPE 不改变向量范数，但不同位置使用不同的 $R_i$。因此完整
post-RoPE Key 二阶矩为

$$
C_K
=
\frac{1}{N}
\sum_{i=1}^{N}
R_i\bar k_i\bar k_i^\mathsf TR_i^\mathsf T.
$$

它一般不能写成某个固定旋转对 pre-RoPE 协方差的共轭变换。即使所有
$\bar k_i$ 都接近同一个低秩子空间，不同位置的旋转也可能扩展 post-RoPE
协方差的秩，或者使 dominant subspace 随位置段变化。

同理，位置 $t$ 的 query 与历史位置 $i$ 的真实 score 为

$$
s_{t,i}
=
\frac{
\bar q_t^\mathsf T
R_t^\mathsf TR_i
\bar k_i
}{\sqrt d}.
$$

真正影响检索的是相对旋转 $R_t^\mathsf TR_i$，而不是单独的 Key 语义空间。
这解释了为什么不能把 PCA 的每个奇异向量解释成固定语义方向；其中可能混合了
内容与相对位置信息。可稳定利用的是整个 dominant score subspace，而不是某个
边界奇异向量的语义。

记首段估计的二阶矩为

$$
\widehat C_{K,L}
=
\frac{1}{m_L}
\sum_{i\in\mathcal I_L}
k_ik_i^\mathsf T,
$$

其中 $\mathcal I_L$ 是长度 $L$ 前缀内 stride 采样的位置。首段 basis 能否代表
完整历史，取决于

$$
\Delta_L
=
\|C_K-\widehat C_{K,L}\|_2
$$

以及 rank 边界 eigengap。Davis--Kahan 型界给出

$$
\|\sin\Theta(P_r,\widehat P_{r,L})\|_2
\lesssim
\frac{\Delta_L}{\lambda_r(C_K)-\lambda_{r+1}(C_K)}.
$$

当前 rank-48 边界 eigengap 很小，因此该最坏情况界不会很紧。更有说服力的证据
是直接测量不同 $L$ 下的完整历史 QK fidelity、score cosine 和 attention mass。
这也是 512/1K/2K/4K/8K prefix 消融的数学动机，而不是普通调参。

---

## 5. PCA 对什么 query 分布是最优的

设 query 的二阶矩为

$$
C_q=\mathbb E[qq^\mathsf T].
$$

对 softmax 有效分数使用中心化 Key

$$
\widetilde K=HK.
$$

对任意 rank-$r$ 正交投影 $P$，期望 QK 重构误差为

$$
\mathcal E^\circ(P)
=
\mathbb E_q\left[
\left\|\widetilde K(I-P)q\right\|_2^2
\right]
=
\operatorname{tr}
\left(
\widetilde K(I-P)C_q(I-P)\widetilde K^\mathsf T
\right).
$$

当 query 在各方向近似各向同性，即

$$
C_q=\alpha I,
$$

上式化为

$$
\mathcal E^\circ(P)
=
\alpha\|\widetilde K(I-P)\|_F^2.
$$

根据 Eckart--Young--Mirsky 定理，$P=P_r^\circ$，即取中心化 Key
$\widetilde K$ 的前 $r$ 个右奇异方向，使该误差最小。

这给出一个重要边界：

> Centered Key-PCA 是各向同性 query 分布下、针对 softmax 有效 QK
> 分数的最优 rank-$r$ 正交投影；对于一般 query 分布，它不保证最优。

若 $C_q$ 与 $\widetilde K^\mathsf T\widetilde K$ 共享特征向量，那么每个
方向的重要性变成

$$
(\sigma_j^\circ)^2\,\mathbb E[(v_j^{\circ\mathsf T}q)^2],
$$

而不只是 $(\sigma_j^\circ)^2$。若改用原始 $K$ 重复上述推导，数学上仍能
得到 raw-score 的最优投影，但该目标可能把维度分配给 softmax 会消掉的 Key
均值方向。冻结生产方法恰好使用未中心化、首 2K sampled Key，因此它与本节
理想最优解之间同时存在“未中心化目标失配”和“首段采样失配”。

生产方法没有学习额外 query 模块，却仍得到较高质量，依赖的是实际模型中 query
与 Key dominant subspace 的经验对齐。这个对齐需要跨任务、跨模型报告，而不能
被当作无条件定理。

---

## 6. Sampled PCA 的扰动边界

设完整二阶矩为 $C_K$，采样估计为 $\widehat C_K$，两者的谱扰动为

$$
E_C=\widehat C_K-C_K.
$$

若第 $r$ 与第 $r+1$ 个特征值之间存在 eigengap

$$
\gamma=\lambda_r-\lambda_{r+1}>0,
$$

则 Davis--Kahan 型子空间扰动界给出

$$
\|\sin\Theta(\widehat V_r,V_r)\|_2
\le
\frac{\|E_C\|_2}{\gamma}.
$$

当前 $\lambda_{48}/\lambda_{49}=1.021$，表明 $\gamma$ 很小，所以这个最坏情况界会很松，不能用于宣称每个边界奇异向量都被准确恢复。

冻结实现还存在一个更具体的有限样本事实：first-2K、stride 32 只产生约

$$
m=\left\lceil\frac{2048}{32}\right\rceil=64
$$

个 Key 样本，却要估计 $r=48$ 维子空间。样本二阶矩的秩最多为 64，且
$r/m=75\%$。因此这里并不处于可以轻易调用“大样本 PCA 一致性”的区域。
标准矩阵 Bernstein 加 Davis--Kahan 的高概率界还需要近似独立同分布、有界
尾部和足够 eigengap；语言序列的等距位置样本不自动满足这些条件。

这不否定方法，但改变了证据标准：不能只报告 sample covariance 的 explained
variance，而必须直接测该 basis 对完整历史 QK 的 fidelity、score cosine、
subspace overlap 和最终 attention mass。前缀 512/1024/2048/4096/8192 的
纯数值消融也由此成为必要实验，而不是普通调参。

但实验观察到两个更直接的结果：

| 对比 | 4% 候选保留的 full-attention mass |
|---|---:|
| Full SVD48 FP32 | 90.49% |
| Sampled PCA48 FP32 | 90.38% |

两者只差 0.11 个百分点。这说明在当前任务上，检索质量对边界子空间的旋转不敏感。论文应报告这种 downstream subspace stability，而不是依赖松弛的 eigengap 最坏情况界。

---

## 7. INT4 是主子空间内的量化扰动

在 PCA 坐标中定义

$$
z_i=V_r^\mathsf Tk_i,\qquad u=V_r^\mathsf Tq.
$$

生产实现使用分组 log-scale INT4 Key 和 INT8 query：

$$
\widehat z_i=z_i+e_{k,i},\qquad
\widehat u=u+e_q.
$$

代理分数为

$$
\widehat s_i
=
\frac{\widehat u^\mathsf T\widehat z_i}{\sqrt d}.
$$

相对 PCA FP32 分数的量化扰动为

$$
\delta_i^{Q}
=
\frac{
u^\mathsf Te_{k,i}
+e_q^\mathsf Tz_i
+e_q^\mathsf Te_{k,i}
}{\sqrt d}.
$$

因此精确 QK 与低维量化代理分数之间满足

$$
s_i-\widehat s_i
=
\delta_i^{\mathrm{PCA}}-\delta_i^Q.
$$

对于候选排序，真正相关的是中心化后的误差
$H(s-\widehat s)$。由于 $\|H(s-\widehat s)\|_2\le
\|s-\widehat s\|_2$，下面的逐 token 未中心化误差界仍然成立，但中心化 NRMSE
与 Pearson 更能反映实际检索扰动。

确定性上界为

$$
|s_i-\widehat s_i|
\le
\frac{
\|q_t\|_2\|k_{t,i}\|_2
+\|u\|_2\|e_{k,i}\|_2
+\|e_q\|_2\|z_i\|_2
+\|e_q\|_2\|e_{k,i}\|_2
}{\sqrt d}.
$$

此前 full-history sampled-PCA48 诊断实测：

| 指标 | 结果 |
|---|---:|
| PCA 尾部 score energy / 精确 score energy | 10.63% |
| INT4 score error energy / 精确 score energy | 2.13% |
| PCA 尾误差与 INT4 误差的平均 cosine | 0.00073 |
| full-history sampled 代理分数与精确 QK 的 Pearson | 0.9338 |

平均 cosine 接近 0 表明两类误差没有观察到系统性同向叠加，但这是一项经验统计，不是独立性证明。量化误差也不应被称为“第二个语义尾部”；它是 dominant subspace 内的数值扰动。

这些数值不能直接冒充冻结实现的 production 数值，因为 production 使用 first-2K
basis。production-aligned 结果把三个误差项分开后得到：

| 代理阶段 | 中心化 score NRMSE | 中心化 Pearson |
|---|---:|---:|
| first-2K PCA48 FP32 | 0.6177 | 0.7771 |
| first-2K PCA48 + INT4 K + INT8 Q | 0.6265 | 0.7712 |

量化新增的 score error energy 约为 exact score energy 的 1.01%，prefix/PCA
误差与 INT4 误差 cosine 为 0.00042。也就是说，生产总误差主要在 basis，
INT4/INT8 只造成较小增量；这与候选 mass 从 86.63% 轻微降到 86.47% 一致。

### 7.1 CountCap 是有偏谱估计，不是无偏随机估计

忽略量化时，CountCap 的代理分数为

$$
\widehat s_P(q,k)=\frac{q^\mathsf TPk}{\sqrt d}.
$$

对固定 $q,k$，其误差是确定性的：

$$
\operatorname{bias}_P(q,k)
=
\widehat s_P-s
=
-\frac{q^\mathsf T(I-P)k}{\sqrt d}.
$$

所以 CountCap 没有逐 token 无偏保证。若把自然 query 看成二阶矩为
$C_q=\mathbb E[qq^\mathsf T]$ 的随机变量，则对固定 Key $k_i$，

$$
\mathbb E_q\left[
(\widehat s_P-s)^2
\right]
=
\frac{1}{d}
k_i^\mathsf T(I-P)C_q(I-P)k_i.
$$

对全部 Key 求和得到

$$
\frac{1}{d}
\operatorname{tr}\left(
K(I-P)C_q(I-P)K^\mathsf T
\right),
$$

这正是第 5 节的期望 QK 重构误差。

随机旋转类代理可以具有

$$
\mathbb E_R[\widetilde s_R]=s,
\qquad
\operatorname{MSE}(\widetilde s_R)=
\operatorname{Var}_R(\widetilde s_R),
$$

而谱投影的 MSE 主要来自截断 bias 与量化误差。两者没有无条件优劣关系：
当自然 query 与 Key dominant subspace 对齐时，CountCap 的 bias 平方可能小于
随机估计方差；当 query 专门落在谱尾方向时，随机无偏估计更稳健。

这也决定了论文边界：CountCap 不应声称支持严格校准的 adaptive top-$p$，
而应证明其有偏代理在当前数据分布上具有较小的 QK fidelity 损失、边界带 mass
和最终输出扰动。

---

## 8. 精确 top-k 不稳定，但 attention 可以稳定

令精确分数与代理分数满足

$$
\widehat s=s+\varepsilon.
$$

若希望精确 top-$k$ 集合完全不变，一个充分条件是

$$
s_{(k)}-s_{(k+1)}
>
2\|\varepsilon\|_\infty.
$$

长上下文中大量低权重 token 的边界分数很密集，这个条件经常不成立。此前
full-history sampled 代理在 4% 预算下的 exact top-4% recall 为 68.99%，因此
不能把方法解释为“恢复了精确 top-k”。production-aligned 诊断进一步确认：
first-2K basis 加 INT4/INT8 后的 exact top-4% recall 为 47.34%，但 mass-weighted
recall 为 92.38%，说明集合指标与真正重要的概率质量明显分离。

但是 softmax 关注的是概率质量，不是集合中每个 token 是否相同。令

$$
p_i=\frac{\exp(s_i)}{\sum_j\exp(s_j)}.
$$

若对所有 token 有 $|\varepsilon_i|\le \epsilon$，则精确与代理 softmax 概率满足

$$
e^{-2\epsilon}
\le
\frac{\widehat p_i}{p_i}
\le
e^{2\epsilon}.
$$

更紧的写法应使用对常数平移不敏感的 score-error range：

$$
R(\varepsilon)=\max_i\varepsilon_i-\min_i\varepsilon_i.
$$

取
$c=(\max_i\varepsilon_i+\min_i\varepsilon_i)/2$，则
$\|\varepsilon-c\mathbf1\|_\infty=R(\varepsilon)/2$，而
$\operatorname{softmax}(\widehat s-c\mathbf1)=
\operatorname{softmax}(\widehat s)$。因此

$$
\boxed{
e^{-R(\varepsilon)}
\le
\frac{\widehat p_i}{p_i}
\le
e^{R(\varepsilon)}
},
$$

并由 Hoeffding lemma 得到

$$
\boxed{
D_{\mathrm{KL}}(p\|\widehat p)
\le
\frac{R(\varepsilon)^2}{8}
}.
$$

该界在最坏情况下仍可能很松，但它说明 score 扰动对概率的影响由 score 误差幅度控制，而不是由 top-k 集合的离散变化直接决定。

实测 4% 候选下：

| 候选选择方法 | Exact top-4% recall | 保留 full-attention mass |
|---|---:|---:|
| Exact QK | 100.00% | 91.45% |
| Full SVD48 FP32 | 73.31% | 90.49% |
| Full-history sampled PCA48 FP32 | 71.88% | 90.38% |
| Production first-2K PCA48 FP32 | 48.00% | 86.63% |
| Production first-2K PCA48 + INT4 K + INT8 Q | 47.34% | 86.47% |
| Production + 256 点 sampled threshold | 46.65% | 86.22% |

生产代理的集合 recall 相对 Exact QK 下降 52.66 个百分点，但 attention mass
相对同预算 Exact QK 下降 4.98 个百分点。说明被交换的 token 大量位于 softmax
边界附近；集合 recall 单独看会显著夸大其对 attention 输出的影响。与此同时，
4.98 个百分点并非零误差，仍需通过 logit、PPL 和端到端任务结果验证下游稳定性。

### 8.1 代理 top-k 的确定性 mass 下界

令 $S^\star$ 是精确分数 $s$ 的 top-$k$ 集合，$\widehat S$ 是代理分数
$\widehat s=s+\varepsilon$ 的 top-$k$ 集合。如果

$$
\|\varepsilon\|_\infty\le\epsilon,
$$

那么对每个被遗漏的 $i\in S^\star\setminus\widehat S$，都可以与一个新增的
$j\in\widehat S\setminus S^\star$ 配对，并且由代理排序可得

$$
\widehat s_j\ge\widehat s_i
\quad\Longrightarrow\quad
s_j\ge s_i-2\epsilon.
$$

因此

$$
\sum_{j\in\widehat S}\exp(s_j)
\ge
e^{-2\epsilon}
\sum_{i\in S^\star}\exp(s_i).
$$

记精确 top-$k$ 的遗漏 mass 为 $\eta^\star$，代理 top-$k$ 的遗漏 mass 为
$\widehat\eta$，则得到

$$
\boxed{
1-\widehat\eta
\ge
e^{-2\epsilon}(1-\eta^\star)
}
$$

以及

$$
\boxed{
\widehat\eta
\le
1-e^{-2\epsilon}(1-\eta^\star)
}.
$$

这个结论不要求两个 top-$k$ 集合相同。它说明代理检索的损失可以拆成两部分：
有限预算本身造成的 $\eta^\star$，以及代理分数误差造成的附加项。全局
$\ell_\infty$ 界通常会被极少数异常 token 主导，因此数值上可能较松，但它给出了
完整的确定性推导。

### 8.2 attention-mass 加权的 crossing risk

为了描述实际中“集合 recall 一般，但 mass 很高”的现象，考虑精确 top-$k$
中的 token $i$ 与集合外 token $j$。定义精确分数间隔

$$
g_{ij}=s_i-s_j\ge0
$$

和 pairwise 代理误差

$$
\xi_{ij}=\varepsilon_j-\varepsilon_i.
$$

只有当

$$
\xi_{ij}\ge g_{ij}
$$

时，$j$ 才可能越过 $i$。如果条件于当前 Q/K，$\xi_{ij}$ 是零均值
sub-Gaussian 变量，方差代理为 $\nu_{ij}^2$，则

$$
\Pr(j\text{ 越过 }i)
\le
\exp\left(
-\frac{g_{ij}^2}{2\nu_{ij}^2}
\right).
$$

由 union bound，代理候选遗漏的期望 attention mass 满足

$$
\mathbb E[\widehat\eta]
\le
\eta^\star
+
\sum_{i\in S^\star}
p_i
\min\left\{
1,\,
\sum_{j\notin S^\star}
\exp\left(
-\frac{g_{ij}^2}{2\nu_{ij}^2}
\right)
\right\}.
$$

这里第二项不是普通的 top-$k$ crossing 数，而是用精确 attention 概率 $p_i$
加权的 crossing risk。边界附近的大量低概率 token 即使频繁换位，对该上界的
贡献仍然很小；高概率 token 通常具有更大间隔，越界概率也更低。这解释了此前
full-history sampled 诊断中 68.99% 集合 recall 仍能保留 90.17% attention mass
的现象。

该概率界依赖“误差近似零均值 sub-Gaussian”这一经验假设，论文中必须报告
$\xi_{ij}$ 的均值、尾部分布以及按 $p_i$ 加权的实际 crossing risk，不能把它写成
无条件定理。PCA 尾误差与 INT4 误差 cosine 接近零，只支持“未观察到系统性同向
叠加”，并不自动证明二者独立。

### 8.3 只有阈值附近的 token 才可能被代理误差交换

令精确 top-$k$ 的边界为 $\tau=s_{(k)}$，并假设
$\|\varepsilon\|_\infty\le\epsilon$。对任意遗漏 token
$i\in S^\star\setminus\widehat S$，存在新增 token
$j\in\widehat S\setminus S^\star$ 满足

$$
\widehat s_j\ge\widehat s_i.
$$

由于 $s_i\ge\tau$ 且 $s_j\le\tau$，可以推出

$$
\tau\le s_i\le\tau+2\epsilon,
\qquad
\tau-2\epsilon\le s_j\le\tau.
$$

所以代理 top-$k$ 与精确 top-$k$ 的对称差必定位于边界带内：

$$
\boxed{
S^\star\triangle\widehat S
\subseteq
\{i:|s_i-\tau|\le2\epsilon\}
}.
$$

进一步，

$$
\widehat\eta-\eta^\star
=
\sum_{i\in S^\star\setminus\widehat S}p_i
-
\sum_{j\in\widehat S\setminus S^\star}p_j
\le
\sum_{\substack{i\in S^\star\\s_i\le\tau+2\epsilon}}p_i.
$$

这个式子比全局 $\ell_\infty$ mass 界更有解释力：真正危险的不是所有
score 误差，而是精确 top-$k$ 边界带中承载了多少 attention mass。

### 8.3.1 从 QK 尾部 Frobenius 能量到 top-k 交换数量

奇异谱推导直接控制的是 $\ell_2/Frobenius$ 误差，而上一节的边界带结论使用
$\ell_\infty$ 误差。二者之间可以建立一个不需要随机性假设的桥梁。

对单个 query，仍记精确 top-$k$ 集合为 $S^\star$、代理集合为
$\widehat S$、精确边界为 $\tau=s_{(k)}$。任取 $\gamma>0$，定义

$$
\mathcal B_\gamma
=
\{i:|s_i-\tau|\le\gamma\},
\qquad
\mathcal L_\gamma
=
\{i:|\varepsilon_i|\ge\gamma\}.
$$

将每个 false negative
$i\in S^\star\setminus\widehat S$ 与一个 false positive
$j\in\widehat S\setminus S^\star$ 一一配对。如果二者均不在
$\mathcal B_\gamma$，则

$$
s_i-s_j>2\gamma.
$$

另一方面，代理排序要求

$$
\widehat s_j\ge\widehat s_i
\quad\Longrightarrow\quad
\varepsilon_j-\varepsilon_i\ge s_i-s_j.
$$

因此该配对中至少一个 token 必须属于 $\mathcal L_\gamma$。每个配对要么消耗
一个边界带 token，要么消耗一个大误差 token，于是

$$
|S^\star\triangle\widehat S|
\le
2\left(
|\mathcal B_\gamma|+|\mathcal L_\gamma|
\right).
$$

再由

$$
|\mathcal L_\gamma|
\le
\frac{\|\varepsilon\|_2^2}{\gamma^2},
$$

得到确定性上界

$$
\boxed{
|S^\star\triangle\widehat S|
\le
2|\mathcal B_\gamma|
+
\frac{2\|\varepsilon\|_2^2}{\gamma^2}
}.
$$

对 $M$ 个 query 的 score error matrix
$E=S-\widehat S$ 求平均，记第 $t$ 个 query 的边界带为
$\mathcal B_{\gamma,t}$，则

$$
\boxed{
\frac{1}{MN}
\sum_{t=1}^{M}
|S_t^\star\triangle\widehat S_t|
\le
\frac{2}{MN}
\sum_{t=1}^{M}
|\mathcal B_{\gamma,t}|
+
\frac{2\|E\|_F^2}{MN\gamma^2}
}.
$$

这正是 QK 奇异谱与候选交换之间缺失的连接：

1. QK 尾部奇异值衰减和 Key-PCA 最优性差距控制 $\|E\|_F^2$；
2. top-$k$ 边界的 score 密度控制 $|\mathcal B_{\gamma,t}|$；
3. 两者同时小，平均候选交换才有严格保证。

如果边界非常密集，即使 QK 的相对 Frobenius 误差很小，集合 recall 仍可能不高。
该定理只约束交换数量，不直接约束交换 token 承载的 attention mass；后者仍需用
第 8.2 节的 mass-weighted crossing risk 和真实保留 mass 验证。

### 8.4 256 点 sampled-quantile 是另一项独立误差

冻结实现不是对全部代理分数做精确 top-$k$。它在历史位置上取 256 个等距
midpoint 样本，用样本分位点 $\widetilde\tau$ 近似完整代理分数的分位点
$\widehat\tau$，然后扫描全部低比特索引并保留
$\widehat s_i\ge\widetilde\tau$ 的 token。

若实测阈值误差满足

$$
|\widetilde\tau-\widehat\tau|\le\rho,
$$

则相对精确 QK top-$k$ 的交换带扩展为

$$
\boxed{
|s_i-\tau|\le2\epsilon+\rho
}.
$$

这里不能直接套用独立同分布采样下的 DKW 界，因为 production 使用确定性的
等距位置样本，而语言序列分数沿位置并非独立同分布。严谨实验应直接报告：

1. $\rho=|\widetilde\tau-\widehat\tau|$；
2. sampled threshold 导致的实际选中比例均值和尾部；
3. 超过 candidate capacity 的 head 比例；
4. 阈值候选保留的 full-attention mass。

即使先采用理想化的 i.i.d. 近似，固定 256 点在很小目标比例下也会产生明显的
相对 rank 波动。假设分数连续且采样位置可视为从完整位置分布独立抽样。经概率
积分变换后，经验 $(1-f)$ 分位数对应的 population tail fraction
$\widehat f$ 近似服从 order-statistic 的 Beta 分布，并有

$$
\operatorname{Std}(\widehat f)
\approx
\sqrt{\frac{f(1-f)}{m+2}}.
$$

因此实际 token 数 $\widehat B=N\widehat f$ 的标准差近似为

$$
\boxed{
\operatorname{Std}(\widehat B)
\approx
N\sqrt{\frac{f(1-f)}{m+2}},
\qquad
\frac{\operatorname{Std}(\widehat B)}{fN}
\approx
\sqrt{\frac{1-f}{(m+2)f}}
}.
$$

当 $m=256$ 时：

| 历史长度 | 目标比例 $f$ | 目标 $B$ | 近似 token 标准差 | 相对目标标准差 |
|---:|---:|---:|---:|---:|
| 32K | 4% | 1280 | 390 | 30.5% |
| 64K | 2% | 1280 | 558 | 43.6% |
| 128K | 1% | 1280 | 793 | 61.9% |

另一个更直观的量是样本中期望落入目标尾部的点数 $mf$：32K、64K、128K
分别只有 10.24、5.12、2.56 个。单个 order-statistic 台阶约对应
$1/(m+1)=0.389\%$ 的 population fraction；在 128K 的 1% 目标下，一个台阶
已经等于目标比例的 38.9%。

如果希望理想化相对标准差不超过 $\varepsilon$，上式要求

$$
m
\gtrsim
\frac{1-f}{f\varepsilon^2}-2.
$$

例如取 $\varepsilon=25\%$，32K/4%、64K/2%、128K/1% 至少约需
382/782/1582 个样本。这不是要求立即修改冻结方法，而是说明固定
`sample_count=256` 不能在长度增长后维持相同的分位数相对精度。

生产采样不是 i.i.d.，但 midpoint 结构本身可以得到另一个确定性条件界。
先假设 $N$ 可被 $m$ 整除、代理分数无 ties，并把历史位置分成 $m$ 个等长区间，
每段取 midpoint。对固定阈值 $t$，令

$$
g_t(i)=\mathbf 1\{\widehat s_i\ge t\},
$$

并假设超过阈值的位置沿序列构成至多 $C_t$ 个连续区间。只有包含这些区间
边界的采样段可能使 midpoint 指示值与段内平均不同，受污染的段至多
$2C_t$ 个。因此完整历史与 midpoint 样本的 tail fraction 满足

$$
\boxed{
\left|
\frac{1}{N}\sum_{i=1}^{N}g_t(i)
-
\frac{1}{m}\sum_{j=1}^{m}g_t(i_j^{\mathrm{mid}})
\right|
\le
\frac{2C_t}{m}
}.
$$

当 $N$ 不能整除 $m$ 时，只额外出现 $O(m/N)$ 的分段权重误差。若
$\widetilde\tau$ 是样本分位点，则在无 ties 条件下，样本选中比例与目标 $f$
最多相差一个 order-statistic 台阶 $1/m$，所以

$$
\left|\widehat f_{\mathrm{full}}(\widetilde\tau)-f\right|
\le
\frac{2C_{\widetilde\tau}+1}{m}
+
O\!\left(\frac{m}{N}\right).
$$

这说明 midpoint 采样何时可靠：高分 token 必须在位置上形成少量连续区域。
若高分 token 高度碎片化，$C_t$ 可随 $N$ 增长，该界就失效。事实上不存在
无条件保证：可以把所有采样 midpoint 的分数设低、未采样位置设高，使 256
个样本完全看不到真实高分尾部。因此必须同时报告阈值附近 exceedance run 数、
实际候选数量和 attention mass，不能只依赖固定样本数。

这不是 production 的严格概率定理，因为等距位置分数具有相关性和位置漂移；
但它揭示了固定样本数与长度封顶预算之间的结构性矛盾：$f$ 越小，候选数量相对
目标越不稳定。最终 64K 实测中，目标 1280 token/head，而单 head 实际范围为
86--3851，模型均值为 1467--1561；128K 范围进一步扩大到 13--6411，模型均值
为 1440--1500。这个结果与上述方差量级一致，说明必须同时报告均值、p95、
极值和质量，不能只写“目标 1280”。

因此完整的候选误差账本是：

$$
\text{full QK}
\rightarrow
\text{first-2K PCA48}
\rightarrow
\text{INT4/INT8 score}
\rightarrow
\text{256 点阈值}
\rightarrow
\text{capacity 截断}.
$$

前四项可以用纯数值诊断逐项隔离；最后一项必须以真实 kernel 的实际选中预算和
overflow 统计为准。

LongBench m4 实际审计得到 Llama/Qwen 的平均目标比例
7.116%/7.063%，真实消费 7.263%/7.262%，实际 token/head 均值
420.8/427.2，overflow head 比例仅 0.0218%/0.0316%。冻结的
`qprojscan/qkvfused` 是异步路径，overflow head 由 CUDA kernel 在 capacity
处截断，host-side proxy top-k fallback 为 0%。因此该极小的 capacity 截断是
误差账本中的最后一项，但不是 Full Attention 回退。

---

## 9. 从遗漏 mass 到 attention 输出的严格定理

令代理分数选出的候选集合为 $C$，其遗漏的精确 full-attention mass 为

$$
\eta=1-\sum_{i\in C}p_i.
$$

候选内使用原始 Q/K 重新计算并归一化：

$$
\widetilde p_i
=
\frac{\exp(s_i)}
{\sum_{j\in C}\exp(s_j)}
=
\frac{p_i}{1-\eta},
\qquad i\in C.
$$

将 $\widetilde p$ 在候选外补零，则有精确恒等式

$$
\|p-\widetilde p\|_1=2\eta.
$$

进一步令

$$
o=\sum_i p_iv_i,\qquad
\widetilde o=\sum_{i\in C}\widetilde p_iv_i.
$$

定义候选内外的条件 Value 均值：

$$
o_C=\sum_{i\in C}\frac{p_i}{1-\eta}v_i,\qquad
o_{\bar C}=\sum_{i\notin C}\frac{p_i}{\eta}v_i.
$$

则

$$
o=(1-\eta)o_C+\eta o_{\bar C},
\qquad
\widetilde o=o_C,
$$

所以

$$
\boxed{
\|o-\widetilde o\|_2
=
\eta\|o_{\bar C}-o_C\|_2
\le
\eta\,\operatorname{diam}(V)
}.
$$

这不是近似式，而是对任意候选集合成立的严格关系。它也说明：

- PCA48/INT4 只通过候选集合影响结果；
- 候选内精确 Q/K/V 计算消除了代理分数误差继续进入 softmax 的路径；
- 真正需要控制的是遗漏 mass $\eta$ 与候选内外 Value 的差异。

全部 6400 个真实探针 case 都满足该输出界。

---

## 10. 多头与输出投影

对同一层的第 $h$ 个 query head，记遗漏 mass 为 $\eta_h$，Value 直径为 $D_h$，则

$$
\|\Delta o_h\|_2\le \eta_hD_h.
$$

多头输出拼接后：

$$
\|\Delta o_{\mathrm{cat}}\|_2
\le
\left(
\sum_h\eta_h^2D_h^2
\right)^{1/2}.
$$

经过输出投影 $W_O$ 写入 residual stream：

$$
\boxed{
\|\Delta x_\ell\|_2
\le
\|W_{O,\ell}\|_2
\left(
\sum_h\eta_{\ell,h}^2D_{\ell,h}^2
\right)^{1/2}
}.
$$

这是从 per-head 检索质量到单层 residual 扰动的条件上界。只看 attention output 的相对 L2 不足以判断最终影响；还需要比较

$$
\frac{\|W_{O,\ell}\Delta o_\ell\|_2}
{\|x_{\ell,\mathrm{residual}}\|_2}.
$$

当前第 0 层在 4% 预算下只保留 68.22% attention mass，attention output cosine 为 0.8612，是明确的困难层。它没有造成同等幅度的 PPL 恶化，可能是因为 $W_O$ 投影、residual 相对尺度和后续归一化削弱了扰动，但该解释还需要直接测量，不能只凭最终 PPL 反推。

---

## 11. 多层传播、logit margin 与 PPL

设第 $\ell$ 层以后到最终 logits 的映射在当前轨迹附近具有局部 Lipschitz 常数 $L_{\ell\rightarrow z}$。对多层稀疏误差做逐层替换，可得条件上界

$$
\|\Delta z\|_2
\le
\sum_{\ell=1}^{L}
L_{\ell\rightarrow z}
\|W_{O,\ell}\|_2
\left(
\sum_h\eta_{\ell,h}^2D_{\ell,h}^2
\right)^{1/2}.
$$

该式严格性的前提是相应局部区域内的 Lipschitz 常数有效。深层 Transformer 的全局最坏情况常数通常极松，因此论文中应把它写成条件稳定性定理，并实测局部 logit 扰动。

令 full 与 sparse 的 logit 差为

$$
d=z^{\mathrm{sparse}}-z^{\mathrm{full}},
$$

并定义对 softmax 常数平移不敏感的扰动范围

$$
R(d)=\max_i d_i-\min_i d_i.
$$

若 full 模型当前 top-1 与 top-2 logit margin 为

$$
m=z_{(1)}-z_{(2)},
$$

则 top-1 token 保持不变的充分条件是

$$
\boxed{m>R(d)}.
$$

对任意目标 token $y$，单 token NLL 变化满足

$$
\Delta\operatorname{NLL}_y
=
\log\mathbb E_{i\sim p_{\mathrm{full}}}
\left[\exp(d_i)\right]
-d_y,
$$

所以严格有

$$
\boxed{
|\Delta\operatorname{NLL}_y|\le R(d)
}.
$$

同样利用 Hoeffding lemma，可得 full 到 sparse 的 KL 上界

$$
\boxed{
D_{\mathrm{KL}}
\left(
p_{\mathrm{full}}\|p_{\mathrm{sparse}}
\right)
\le
\frac{R(d)^2}{8}
}.
$$

因此，如果第 $t$ 个评估 token 的 logit 扰动范围为 $R_t$，则整个语料的
PPL 比值满足

$$
e^{-\overline R}
\le
\frac{\operatorname{PPL}_{\mathrm{sparse}}}
{\operatorname{PPL}_{\mathrm{full}}}
\le
e^{\overline R},
\qquad
\overline R=\frac{1}{T}\sum_{t=1}^T R_t.
$$

该证书比 $2\|d\|_\infty$ 更紧，并且不会把对所有 logits 同时加上的常数误判为
模型行为变化。32K、每模型 3072 个 token 的配对实验得到：

| 模型 | top-1 agreement | margin 可认证比例 | 平均 KL | 平均 NLL 变化 | 对应 PPL 倍数 |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 94.69% | 40.27% | 0.01554 | +0.01098 | 1.0110x |
| Qwen3-4B | 91.73% | 28.00% | 0.03201 | +0.00682 | 1.0068x |

两个模型的 NLL/KL 理论界通过率均为 100%，且已通过 margin 条件认证的 token
没有出现 top-1 翻转。未认证不等于必然翻转；它只表示当前充分条件不足以证明稳定。

已有 PPL 结果更直接：

$$
\Delta\operatorname{NLL}
=
\log\frac{8.5064}{8.3930}
=0.01342\ \text{nat/token},
$$

即 32K 体育与医学 targeted PPL 只增加 1.35%。

64K/128K 多主题连续文本压力测试给出了更严格的反例。每个模型和长度包含
4 个 Full/CountCap 配对窗口、共 1024 个目标 token：

| 模型 | 长度 | 平均 $\Delta\mathrm{NLL}$ | PPL 倍数 | PPL 质量保持率 | 实际 attention 消费 | decode 加速 | 含 prefill 协议加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 64K | +0.02478 | 1.0251x | 97.55% | 2.439% | 2.64x | 1.316x |
| Llama-3.1-8B | 128K | +0.06979 | 1.0723x | 93.26% | 1.172% | 4.59x | 1.240x |
| Qwen3-4B | 64K | +0.09028 | 1.0945x | 91.37% | 2.293% | 2.77x | 1.395x |
| Qwen3-4B | 128K | +0.14501 | 1.1560x | 86.50% | 1.125% | 4.12x | 1.261x |

其中 Qwen3-4B 64K 的

$$
\overline{\Delta\operatorname{NLL}}=0.09028,
\qquad
\frac{\operatorname{PPL}_{\mathrm{sparse}}}
{\operatorname{PPL}_{\mathrm{full}}}
=1.09448.
$$

逐 token 分布为：

| 模型 | 长度 | NLL 差中位数 | p95 / p99 | NLL 变差比例 | $\Delta\mathrm{NLL}>1$ / $>2$ | 最大 10% 正向误差占比 |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 64K | +0.0165 | +0.369 / +0.708 | 66.31% | 0.29% / 0.00% | 42.93% |
| Llama-3.1-8B | 128K | +0.0404 | +0.681 / +1.278 | 69.53% | 1.95% / 0.29% | 45.00% |
| Qwen3-4B | 64K | +0.0197 | +0.926 / +2.544 | 63.38% | 4.69% / 1.27% | 49.31% |
| Qwen3-4B | 128K | +0.0403 | +1.329 / +2.634 | 65.04% | 7.52% / 2.05% | 47.26% |

所以长文本退化既不是所有 token 的均匀平移，也不是仅由一个异常 token
造成；它由大量小扰动和少量重尾误差共同构成。最差 token 包含罕见词片段、
专名、长空白及格式符，未呈现单一语义类别。该结果否定
“centered QK 尾能量平均较小就能保证任意位置 PPL 不变”，但不否定前述
条件链：它说明部分长文本位置的阈值边界、prefix basis 或下游 logit margin
不满足紧致稳定条件。Llama 明显比 Qwen 稳定也表明该结论具有模型依赖性。
LongBench 任务分数与连续文本 PPL 必须分别报告。

严格 full-prompt 7.5K LongBench16（每模型 1600 个 Full/CountCap 配对）
进一步说明二者确实衡量不同性质：

| 模型 | Full Macro | CountCap Macro | 保持率 | Macro 差值 95% CI | 平均实际目标比例 | online/token speed |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 0.44558 | 0.44609 | 100.11% | [-0.00228, +0.00321] | 7.40% | 0.896x |
| Qwen3-4B | 0.40988 | 0.41051 | 100.15% | [-0.00288, +0.00439] | 7.35% | 0.777x |
| Qwen2.5-7B | 0.44007 | 0.43385 | 98.59% | [-0.01023, -0.00264] | 7.35% | 0.881x |

因此当前证据支持 Llama/Qwen3 上“短 LongBench 自由生成分数统计等价”，
但 Qwen2.5 上存在显著的约 1.41% 相对下降；三者短序列上都未兑现速度。
这些结果仍不覆盖 64K/128K 连续文本的 token-level 语言建模稳定性。

---

## 12. 为什么无法得到无条件答案不变证明

构造一个尾部对齐 query：

$$
q=v_j,\qquad j>r.
$$

则

$$
P_rq=0,
$$

PCA 代理分数对该 query 完全丢失，而精确分数为

$$
s=\frac{\sigma_j u_j}{\sqrt d},
$$

一般不为零。如果该方向对应一个决定答案的稀有证据 token，截断可能改变候选、attention 和最终答案。

所以严谨结论是：

> 当前模型和自然输入中的 Key 谱高度集中，实际 query 很少同时对齐高影响的尾部 Key 方向；在这种经验条件下，PCA48+INT4 候选保留了接近 Exact-QK 稀疏上界的 attention mass，并通过候选内精确 Q/K/V 计算把数值近似限制在候选选择阶段。

这比“尾部没有语义”更准确，也更经得住审稿。

---

## 13. 当前证据链完成到哪里

| 理论环节 | 需要的量 | 当前证据 |
|---|---|---|
| 谱集中 | 中心化有效秩、能量、奇异值衰减 | 三模型 200 个 topic-layer-KV-head 完成；中心化有效秩 4.72/5.43/6.29，最优 rank-48 保能 99.09%/98.52%/98.80% |
| SVD/PCA 等价 | 同一 uncentered 子空间 | 数学等价；实测 mass 只差 0.11pp |
| sampled-full basis 稳定 | 子空间重合、下游 mass | 91.50% overlap；已完成 |
| first-2K basis 稳定 | 中心化 QK fidelity、score cosine、prefix 长度消融 | 三模型完成；fidelity 62.39%/59.07%/60.37%，cosine 0.7953/0.7609/0.7864，说明它是主要失配项 |
| INT4 数值扰动 | score energy、相关性 | production-aligned 完成；INT4 增量约占 exact score energy 1.01%，总 Pearson 主要受 prefix basis 限制 |
| 候选质量 | exact top-k recall、保留 mass | production INT4/INT8 为 47.34%、86.47%；mass-weighted recall 92.38% |
| sampled threshold | 阈值误差、实际预算、overflow | 256 点 midpoint 诊断与 LongBench m4 审计完成；真实消费 7.263%/7.262%，overflow head 0.0218%/0.0316%，host fallback 0% |
| attention 输出 | mass-output 恒等式与界 | 严格推导；6400/6400 case 满足 |
| residual 写入 | $\|W_O\Delta o\|/\|x\|$ | 尚缺 |
| 最终 logits | KL、top-1 agreement、margin flip | 双模型各 3072 token 完成；top-1 94.69%/91.73%，NLL +0.01098/+0.00682 |
| 最终语言质量 | PPL | 32K targeted PPL +1.35%；64K/128K 质量保持率为 Llama 97.55%/93.26%、Qwen 91.37%/86.50%，证明稳定性不是无条件且具有模型依赖 |
| 自由生成质量 | LongBench/RULER | 严格 full-prompt 7.5K LongBench16 每模型 1600 对已完成：Llama/Qwen3/Qwen2.5 保持率 100.11%/100.15%/98.59%；前两者 CI 跨 0，Qwen2.5 CI 为 [-1.023pp,-0.264pp]；短序列 online/token 仅 0.896x/0.777x/0.881x |

---

## 14. 论文中建议使用的定理表述

可以把核心理论写成下面三条，而不要声称端到端无条件等价。

**命题 1：中心化谱最优性。** 在各向同性 query 二阶矩下，中心化 Key
$\widetilde K=HK$ 的前 $r$ 个右奇异向量，给出期望 softmax 有效 QK 分数
平方误差最小的 rank-$r$ 正交投影。冻结实现使用未中心化、首 2K 采样子空间，
因此其相对该最优解的差距必须实验测量，不能由该命题直接消除。

**命题 2：候选 attention 稳定性。** 对任意候选集合 $C$，若其遗漏 full-attention mass 为 $\eta$，则候选内使用精确 Q/K/V 重算后的 attention 分布满足 $\|p-\widetilde p\|_1=2\eta$，输出满足 $\|o-\widetilde o\|_2\le\eta\operatorname{diam}(V)$。

**命题 3：条件 logit 稳定性。** 令 full 与 sparse 的最终 logit 差为 $d$，扰动范围为 $R(d)=\max d-\min d$。当 full top-1 margin 大于 $R(d)$ 时预测 token 不变；任意目标 token 的 NLL 变化不超过 $R(d)$，且 $D_{\mathrm{KL}}(p_{\mathrm{full}}\|p_{\mathrm{sparse}})\le R(d)^2/8$。

三条命题共同解释了为什么方法无需恢复完全相同的 top-k token，也能保持最终质量；实验负责验证自然输入是否满足小遗漏 mass、小局部传播和足够 logit margin。

## 相关工作边界

- Loki 已证明 Key 具有低秩结构，并使用离线 PCA 低维分数选择 token：
  <https://arxiv.org/abs/2406.02542>
- LRQK 已直接研究 Q/K 联合低秩分解与低维 QK 检索：
  <https://arxiv.org/abs/2510.23649>
- SVDq 已利用 Key 奇异谱进行 latent-channel 混合精度量化：
  <https://arxiv.org/abs/2502.15304>
- RaBitQCache 已给出随机旋转低比特代理的无偏性和误差界：
  <https://arxiv.org/abs/2606.31519>
- SALS 已在 RoPE-free latent Q/K 空间执行稀疏选择并只重构候选：
  <https://papers.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html>
- Self-Indexing KVCache 已把中心化、低比特压缩 Key 同时用作 top-$k$ 索引：
  <https://ojs.aaai.org/index.php/AAAI/article/view/39988>
- RocketKV 已联合 head/sequence 降维代理与两阶段 KV 驱逐：
  <https://proceedings.mlr.press/v267/behnam25a.html>
- Thin Keys 与 STAR-KV 已分别研究权重 SVD 后的低维 Key、head/block 自适应
  低秩 K/V：
  <https://arxiv.org/abs/2603.04427>，
  <https://arxiv.org/abs/2606.08382>

因此，PCA/SVD 低秩或 QK 奇异值衰减本身都不能作为 CountCap 的独立创新点。
本文推导的作用是解释冻结实现中
`first-2K basis + INT4/INT8 + sampled threshold + direct sparse attention`
的完整误差传播，并明确它在哪些自然输入条件下成立、在哪些尾部 query 上可能失败。

## 关联产物

- 机制分析：`docs/20260726_pca48_int4_delta_theory_and_evidence_zh.md`
- production-aligned 汇总：`results/20260726_pca48_int4_delta_invariance_32k_v4/summary.json`
- 中心化 QK 谱汇总：`results/20260726_qk_matrix_spectrum_multimodel_32k/analysis_centered/summary.json`
- 中心化 QK 谱明细：`results/20260726_qk_matrix_spectrum_multimodel_32k/analysis_centered/qk_spectrum_rows.csv`
- Qwen 64K token 级 NLL 尾部：
  `results/20260726_countcap_final_long_speed_multimodel_4gpu/qwen3_4b/token_nll_tail_64k.json`
- score 明细：`results/20260726_pca48_int4_delta_invariance_32k_v4/score_rows.csv`
- 候选与输出明细：`results/20260726_pca48_int4_delta_invariance_32k_v4/candidate_rows.csv`

---

## 15. 三项理论闭环补充

前缀 PCA 泛化、margin-conditioned candidate preservation 和完整请求级成本模型已在下列文档中单独闭合：

- `docs/20260726_countcap_three_theory_closure_zh.md`

该补充增加了三项此前没有完整覆盖的结果：

1. prefix-basis QK 残差的精确分解，以及不依赖 eigengap 的 PCA excess-risk 定理；
2. uniform range、sampled threshold、boundary mass 与逐 token interval 四类候选保持证书；
3. 从逐 token 计时中分离 lazy index 固定成本与 steady decode 成本，并给出长度和生成步数交叉点。

同时确认两项重要负结果：

- rank-48 Davis--Kahan 界在三个模型的 2K-prefix 诊断上有效比例均为 0%，不能用来声称 prefix 子空间在 spectral norm 下稳定；
- 全局 $L_2$ crossing bound 在 32K targeted trace 上饱和到完整候选预算，严格但数值过松；逐 token norm interval 才提供非空证书。

新 targeted margin 实验使用体育和医学 trace，难度高于本附录前面的平均候选诊断。因此 production sampled top-4% 的总体 mass recall 为 80.48%，低于前文 92.38% 的平均结果。两者对应不同评测分布，应分别报告，不能混为同一个统计量。
