# QKSieve 理论分析与可验证证明

本文档对应论文中的 QKSieve 主方法：

- 完整 FP16 K/V 常驻 GPU；
- 每层、每个 KV head 独立构造 QK-balanced 坐标；
- 128 维被划分为 8 个 16 维 band；
- 每个 band 自动选择 0/1/2/4/8 bit；
- 总索引预算固定为 240 bit/token/KV-head；
- decode 时扫描完整低比特索引，直接取近似分数 top-k；
- 对选中位置使用原始 FP16 Q/K/V 做精确稀疏 attention；
- 不使用采样阈值、精确重排、router、任务规则、recent/sink 保护或 Full fallback。

下面严格区分四类结论：

1. 无条件成立的代数恒等式；
2. 对给定有限样本精确成立的统计恒等式；
3. 需要显式条件的误差上界；
4. 必须由实验验证的分布假设。

## 1. 问题定义

对某一层、某个 query head，完整 attention score 为：

$$
s_i=\frac{q^\top k_i}{\sqrt d},\qquad i=1,\ldots,N.
$$

完整 attention 输出为：

$$
p_i=\frac{\exp(s_i)}{\sum_j\exp(s_j)},\qquad
o=\sum_i p_i v_i.
$$

QKSieve 希望用廉价索引估计所有 score，选出集合 $S$，再对 $S$ 内的原始 FP16 K/V 计算：

$$
\widetilde p_i=
\frac{\exp(s_i)}{\sum_{j\in S}\exp(s_j)},\qquad
\widetilde o=\sum_{i\in S}\widetilde p_i v_i.
$$

因此误差只来自“选中了哪些 token”，不是最终 attention 的数值精度。

## 2. QK-balanced 变换为何不会损失原始 score

### 2.1 构造

分别估计 Key 和 Query 的未中心化二阶矩：

$$
C_k=\frac{1}{m_k}\sum_i k_i k_i^\top,\qquad
C_q=\frac{1}{m_q}\sum_j q_j q_j^\top.
$$

对 Query moment 做固定收缩：

$$
\widetilde C_q=(1-\lambda)C_q+
\lambda\frac{\operatorname{tr}(C_q)}{d}I.
$$

计算：

$$
\widetilde C_q^{1/2}C_k^{1/2}=U\Sigma V^\top.
$$

定义两个不同的变换：

$$
A=\widetilde C_q^{-1/2}U\Sigma^{1/2},\qquad
D=C_k^{-1/2}V\Sigma^{1/2}.
$$

Query 和 Key 分别变换为：

$$
q'=A^\top q,\qquad k'=D^\top k.
$$

### 2.2 定理 1：全维 score 精确保持

只要两个二阶矩正定，则：

$$
AD^\top=I.
$$

证明：

$$
\begin{aligned}
AD^\top
&=\widetilde C_q^{-1/2}U\Sigma^{1/2}
\Sigma^{1/2}V^\top C_k^{-1/2}\\
&=\widetilde C_q^{-1/2}
\left(\widetilde C_q^{1/2}C_k^{1/2}\right)
C_k^{-1/2}\\
&=I.
\end{aligned}
$$

所以对任意 Query 和 Key，而不只是用于估计矩阵的样本：

$$
{q'}^\top k'
=q^\top AD^\top k
=q^\top k.
$$

结论：QK-balanced 不是低秩近似。若不量化、不删维，它对 QK score 的理论误差严格为零。

## 3. 为什么奇异值可以表示“共同的 QK 信息”

### 3.1 定理 2：Query 和 Key 被平衡到相同谱

直接代入可得：

$$
A^\top\widetilde C_qA
=D^\top C_kD
=\Sigma.
$$

也就是说，在变换后的坐标中，Query 与 Key 的二阶矩都是同一个对角矩阵。

若从这两个二阶矩独立取样，则 score 能量为：

$$
\mathbb E[({q'}^\top k')^2]
=\operatorname{tr}(\Sigma^2)
=\sum_{\ell=1}^{d}\sigma_\ell^2.
$$

因此第 $\ell$ 个坐标对 QK score 的二阶贡献由 $\sigma_\ell^2$ 描述。奇异值快速衰减时，前部坐标承载大部分共同 score 能量，尾部坐标的平均影响较小。

这比只对 Key 做 PCA 更贴近检索目标：

- Key-PCA 按 Key 重建能量排序；
- QK-balanced 按 Query 与 Key 的共同 score 能量排序；
- 当 $C_q$ 近似各向同性时，两者的排序接近；
- 当 Query 明显各向异性时，QK-balanced 会把 bit 分配给 Query 实际敏感的方向。

这里可以借助 SVD 的奇异值集中解释现象，但不能把方法描述成普通低秩 SVD：QKSieve 保留完整 128 维坐标，只对不同坐标采用不同 bit。

### 3.2 定理 3：前 $r$ 个 QK-balanced 方向是 score 近似的全局最优子空间

设零均值 Query 和 Key 独立，二阶矩分别为 $C_q,C_k$。考虑所有秩不超过 $r$ 的双线性近似：

$$
q^\top k\approx q^\top B_rk,\qquad \operatorname{rank}(B_r)\le r.
$$

记：

$$
C_q^{1/2}C_k^{1/2}=U\Sigma V^\top.
$$

则最小化期望 score MSE 的全局最优矩阵是：

$$
B_r^\star
=C_q^{-1/2}U_r\Sigma_rV_r^\top C_k^{-1/2}
=A_rD_r^\top,
$$

而最小误差恰好为：

$$
\min_{\operatorname{rank}(B_r)\le r}
\mathbb E\left[(q^\top k-q^\top B_rk)^2\right]
=\sum_{\ell>r}\sigma_\ell^2.
$$

证明从下面的恒等式开始：

$$
\mathbb E[(q^\top(I-B_r)k)^2]
=
\left\|
C_q^{1/2}(I-B_r)C_k^{1/2}
\right\|_F^2.
$$

由于 $C_q,C_k$ 正定，$C_q^{1/2}B_rC_k^{1/2}$ 可以遍历所有秩不超过 $r$ 的矩阵。问题因此等价于对 $C_q^{1/2}C_k^{1/2}$ 求最佳秩 $r$ Frobenius 近似，结论直接来自 Eckart--Young--Mirsky 定理。

这个结果比“前部奇异值较大”更强：在明确的二阶矩和独立性条件下，QK-balanced 前 $r$ 维不是经验上较好，而是所有秩 $r$ 双线性 score 近似中的最优解。Key-PCA 则只对 Key 重构误差最优。

### 3.3 Key-PCA 是 Query 各向同性时的特殊情况

若 $C_q=\alpha I$，并记：

$$
C_k=V\Lambda V^\top,
$$

则：

$$
C_q^{1/2}C_k^{1/2}
=V(\alpha^{1/2}\Lambda^{1/2})V^\top.
$$

此时 QK-balanced 的左右奇异向量都等于 Key-PCA 的特征向量 $V$，而且按
$\sqrt{\alpha\lambda_\ell}$ 排序与按 Key 特征值 $\lambda_\ell$ 排序完全相同。两者只在 Query/Key 两侧使用互相补偿的缩放，仍满足 $AD^\top=I$。

因此：

- 当 Query 对所有方向同等敏感时，QK-balanced 退化为 Key-PCA；
- 当 Query 二阶矩具有明显各向异性时，QK-balanced 才会改变方向和 bit 优先级；
- “Key-PCA 与 QK-balanced 的差距应随 Query anisotropy 增大”是可以直接做实验验证的推论。

## 4. 为什么量化目标应该是 QK error，而不是 Key MSE

设量化后的 Key 为：

$$
\widehat k'=k'-e.
$$

单个 score 的误差为：

$$
{q'}^\top k'-{q'}^\top\widehat k'
={q'}^\top e.
$$

### 4.1 定理 3：有限样本 logit-MSE 恒等式

对任意有限 Query 样本 $\{q'_j\}$ 和任意有限 Key error 样本 $\{e_i\}$，定义：

$$
C'_q=\frac{1}{m_q}\sum_j q'_j{q'_j}^\top,\qquad
C_e=\frac{1}{m_k}\sum_i e_i e_i^\top.
$$

则所有 Query-Key 笛卡尔配对上的平均 score MSE 精确等于：

$$
\frac{1}{m_qm_k}\sum_{j,i}({q'_j}^\top e_i)^2
=\operatorname{tr}(C'_qC_e).
$$

证明只使用：

$$
(x^\top y)^2=\operatorname{tr}(xx^\top yy^\top)
$$

和求和的线性性，不需要零均值、高斯分布或样本独立假设。

若使用 attention 中的归一化 logit $q^\top k/\sqrt d$，上述 MSE 整体再除以 $d$。这是所有 allocation 共享的常数，不会改变最优 bit 分配。

而 Key 重建 MSE 是：

$$
J_K=\operatorname{tr}(C_e).
$$

只有当 $C'_q=\alpha I$ 时，两个目标才对所有量化误差给出相同排序：

$$
J_{QK}=\operatorname{tr}(C'_qC_e)=\alpha J_K.
$$

当 Query 各向异性时，可以构造两个 error covariance，使 Key MSE 偏好方案 1，而 QK MSE 偏好方案 2。因此，低 Key 重建误差既不是低 score 误差的充分条件，也不是必要条件。

## 5. 分 band 自动 bit 分配为何合理

把 128 维分成 8 个 16 维 band。第 $g$ 个 band 使用 $b_g\in\{0,1,2,4,8\}$ bit。

16 维 active band 还需一个 FP16 scale，相当于每维额外 1 bit，因此物理预算为：

$$
R(b)=16\sum_g\left(b_g+\mathbf 1[b_g>0]\right)\le240\text{ bit}.
$$

第 $g$ 个 band 的经验失真为：

$$
D_g(b)=\operatorname{tr}\left(C'_{q,g}C_{e,g}(b)\right).
$$

自动分配求解：

$$
\min_{b_1,\ldots,b_8}\sum_gD_g(b_g)
\quad\text{s.t.}\quad R(b)\le240.
$$

因为只有 8 个 band、5 个 bit 选择和固定的小预算，可以枚举所有可行组合或使用动态规划得到全局最优解，不需要训练 router。

### 5.1 同预算下自动分配的校准目标支配性

记所有满足 240-bit 物理预算的 allocation 集合为 $\mathcal B_R$。由于动态规划遍历了完整可行集合，所以：

$$
\sum_gD_g(b_g^\star)
\le
\sum_gD_g(\bar b_g),
\qquad
\forall\bar b\in\mathcal B_R.
$$

因此，在相同 quantizer 和当前 band 可分目标下，自动 allocation 的校准 QK-MSE 不可能差于固定 4-4-4、4-4-2-1 等任意可行 allocation。这是优化问题本身的确定性保证。

它不等于 held-out Query 上也必然更好。有限 Query 样本误差和真实 Query drift 仍可能改变方案排序，这正是第 6 节的界和独立测试需要解决的问题。

### 5.2 band 可分性的条件和误差

完整 QK MSE 可写成：

$$
\operatorname{tr}(C'_qC_e)
=\sum_g\operatorname{tr}(C'_{q,gg}C_{e,gg})
+\sum_{g\ne h}\operatorname{tr}(C'_{q,gh}C_{e,hg}).
$$

当前 allocator 最小化第一项。若 Query moment 或量化 error moment 在这些 band 上近似块对角，则该目标近似完整目标。

忽略的交叉项满足：

$$
\left|
\sum_{g\ne h}\operatorname{tr}(C'_{q,gh}C_{e,hg})
\right|
\le
\sum_{g\ne h}
\|C'_{q,gh}\|_F\|C_{e,hg}\|_F.
$$

这不是无法验证的假设。实验必须报告“交叉项 / 完整 QK MSE”的比例。

### 5.3 为什么高敏感 band 应自动得到更多 bit

为了看清 bit 分配的方向，先考虑一个理想化的连续高码率模型：

$$
D_g(b_g)=\alpha_g2^{-2b_g},\qquad
\sum_g b_g\le R,\qquad b_g\ge0.
$$

其中 $\alpha_g$ 表示第 $g$ 个 band 对 QK score 的敏感度。KKT 条件给出类似“注水”的闭式解：

$$
b_g^\star=
\left[\frac12\log_2\frac{\alpha_g}{\tau}\right]_+,
\qquad
\sum_gb_g^\star=R.
$$

如果所有 band 都处于激活状态，则：

$$
b_g^\star=
\frac RG+
\frac12\log_2\frac{\alpha_g}{\bar\alpha},
$$

其中 $\bar\alpha$ 是所有 $\alpha_g$ 的几何平均。也就是说，Query 更敏感、score distortion 下降更快的 band 应分到更多 bit；不敏感的 band 会被压到低 bit，甚至 0 bit。

这只是解释性公式，不是实现假设。真实 QKSieve 会显式计算 0/1/2/4/8 bit 的经验 $D_g(b)$，同时计入 FP16 scale 元数据，再由离散 DP 求全局最优。因此，即使有限 bit 误差不严格服从 $2^{-2b}$，生产分配仍然成立。

### 5.4 自动 mixed-bit 相对统一 bit 的理论收益

在同一个坐标系、所有 band 都处于激活状态时，统一分配使用 $b_g=R/G$。其失真与最优 mixed-bit 失真的比值为：

$$
\frac{J_{\mathrm{uniform}}}{J_{\mathrm{mixed}}^\star}
=
\frac{
\frac1G\sum_{g=1}^{G}\alpha_g
}{
\left(\prod_{g=1}^{G}\alpha_g\right)^{1/G}
}
\ge 1.
$$

右边正好是 band 敏感度的“算术平均 / 几何平均”。由算术—几何平均不等式：

- 只有全部 band 的 $\alpha_g$ 完全相同时，统一 bit 才与 mixed-bit 等价；
- 敏感度越不均匀，自动分配的理论优势越大；
- 该优势与总 rate 中共同的 $2^{-2R/G}$ 因子无关。

证明很直接。统一分配的失真是
$2^{-2R/G}\sum_g\alpha_g$；最优解中每个激活 band 的失真都等于
$\tau=(\prod_g\alpha_g)^{1/G}2^{-2R/G}$，因此总失真为 $G\tau$，两者相除即得上式。

这给出一个可证伪实验预测：逐层、逐 head 计算
$\operatorname{AM}(\alpha)/\operatorname{GM}(\alpha)$，它应与“uniform bit 相对 auto mixed-bit 的 held-out QK-MSE 降幅”正相关。该结论只隔离固定坐标系中的 bit 分配收益；FIER 还使用原始坐标，因此正式实验要同时报告 QK-balanced uniform-bit 消融和 FIER 对比，不能把坐标变换收益全部算作 mixed-bit 收益。

该统计已经接入
`src/analyze_fier_qksieve_retrieval_fair_20260728.py`。重新运行后会生成
`heterogeneity.csv`，并在 `summary.json` 中报告 log-Pearson 和
Spearman 相关系数；当前因为 GPU 任务暂停，尚未产生这组新结果。

## 6. 生成期 Query 漂移的理论边界

方法在 prompt 末尾采样 Query，之后冻结变换和 bit 分配。设：

- $C'_{q,\mathrm{cal}}$：校准 Query moment；
- $C'_{q,\mathrm{dec}}$：生成期 Query moment。

对任意固定 Key error covariance：

$$
\left|
\operatorname{tr}(C'_{q,\mathrm{dec}}C_e)
-\operatorname{tr}(C'_{q,\mathrm{cal}}C_e)
\right|
\le
\|C'_{q,\mathrm{dec}}-C'_{q,\mathrm{cal}}\|_{\mathrm{op}}
\operatorname{tr}(C_e).
$$

证明：将 $C_e$ 做特征分解，然后对每个特征方向使用 operator norm 的定义即可。

该式指出两个独立风险：

1. 生成期 Query 分布与 prompt-tail Query 差异很大；
2. Key 量化残差本身很大。

因此长生成和多轮对话实验必须按生成位置报告 Query covariance drift、实际 score MSE 和 attention mass recall。

### 6.1 有限 Query 样本与真实分布漂移必须分开

令 $C'_{q,\mathrm{cal}}$ 为校准 Query 的总体二阶矩，$\widehat C'_{q,m}$ 为 $m$ 个独立校准 Query 的经验二阶矩。定义：

$$
X_j=q'_j{q'_j}^\top-C'_{q,\mathrm{cal}}.
$$

若 $\|X_j\|_{\mathrm{op}}\le L$ 且 $\|\mathbb E X_j^2\|_{\mathrm{op}}\le v$，则 Matrix Bernstein 给出：以至少 $1-\delta$ 的概率，

$$
\begin{aligned}
\|\widehat C'_{q,m}-C'_{q,\mathrm{dec}}\|_{\mathrm{op}}
\le{}&
\sqrt{\frac{2v\log(2d/\delta)}{m}}
+\frac{2L\log(2d/\delta)}{3m}\\
&+\|C'_{q,\mathrm{cal}}-C'_{q,\mathrm{dec}}\|_{\mathrm{op}}.
\end{aligned}
$$

右侧前两项是“只采 8 个 Query 带来的估计误差”，最后一项是“prompt-tail 与真实 decode Query 的分布漂移”。增加 Query 样本只能降低前者，不能修复后者。由于自回归 Query 一般并不独立同分布，该定理不能代替长生成实验，但它明确说明了 Query 数量消融和生成位置漂移实验分别在验证什么。

### 6.2 held-out Query 上的 allocation regret

前面的结论还可以进一步回答一个更直接的问题：校准期选出的 $b^\star$，到真实生成期最多可能比另一个固定方案 $\bar b$ 差多少？

令 $\widetilde J_t(b)$ 为只保留 band 内项的 QK-MSE，$\Gamma_t(b)$ 为跨 band 项，且
$J_t(b)=\widetilde J_t(b)+\Gamma_t(b)$。再定义：

$$
\Delta=C'_{q,\mathrm{dec}}-C'_{q,\mathrm{cal}},
$$

$$
\Omega(b)=
\sum_g
\|\Delta_{gg}\|_{\mathrm{op}}
\operatorname{tr}(C_{e,gg}(b_g)).
$$

则对任意相同物理预算下的固定方案 $\bar b$：

$$
\begin{aligned}
J_{\mathrm{dec}}(b^\star)-J_{\mathrm{dec}}(\bar b)
\le{}&
\Omega(b^\star)+\Omega(\bar b)\\
&+|\Gamma_{\mathrm{dec}}(b^\star)|
+|\Gamma_{\mathrm{dec}}(\bar b)|.
\end{aligned}
$$

证明只需要三步：把生成期目标在 $b^\star$ 和 $\bar b$ 处分别加减校准期目标；利用 $b^\star$ 在校准可分目标上全局最优；最后用 operator norm 和 trace 对两侧漂移项取上界。

这个界给出了清楚、可证伪的结论：

- 若 block-diagonal Query drift 小，且跨 band 项小，自动方案在生成期仍接近同预算最优方案；
- 若 held-out 结果退化，可以把原因拆成 Query drift 和 band 可分近似两部分；
- 右侧所有量都能从无标签的真实 decode Query 与 Key error 中测量；
- 该定理不声称 prompt-tail allocation 在任意分布漂移下仍然最优。

## 7. 从 score 误差到 top-k 保留

记近似 score 为 $\widehat s_i$，误差为：

$$
\delta_i=s_i-\widehat s_i.
$$

### 7.1 Softmax 对公共平移无关，只受误差范围控制

定义误差范围：

$$
R_\delta=\max_i\delta_i-\min_i\delta_i.
$$

令 $p=\operatorname{softmax}(s)$、$\widehat p=\operatorname{softmax}(\widehat s)$，则：

$$
\operatorname{KL}(p\|\widehat p)\le\frac{R_\delta^2}{8},
$$

以及：

$$
\|p-\widehat p\|_1
\le
\min\left\{2,\frac{R_\delta}{2}\right\}.
$$

证明利用：

$$
\operatorname{KL}(p\|\widehat p)
=
\mathbb E_{i\sim p}[\delta_i]
+\log\mathbb E_{i\sim p}[\exp(-\delta_i)],
$$

再应用 Hoeffding 引理和 Pinsker 不等式。这个界只依赖误差范围，而不依赖误差中的公共偏移，符合 softmax 和排序对公共平移不敏感的事实。

QKSieve 最终不会用 $\widehat p$ 聚合 Value，而是只用近似 score 选集合，再用精确 score 计算 attention。因此这个结论用于刻画低比特索引的数值稳定性；最终输出误差仍应使用第 8 节的“遗漏 attention mass”界。

### 7.2 top-k 排序证书

误差的公共平移不影响排序，所以定义：

$$
\bar\delta=\frac{1}{N}\sum_i\delta_i,\qquad
\eta^2=\frac{1}{N}\sum_i(\delta_i-\bar\delta)^2.
$$

有：

$$
\max_i|\delta_i-\bar\delta|\le\sqrt N\,\eta,
$$

从而：

$$
\max_i\delta_i-\min_i\delta_i\le2\sqrt N\,\eta.
$$

设精确 score 从大到小为 $s_{(1)},\ldots,s_{(N)}$。若对某个 $r\le B$：

$$
s_{(r)}-s_{(B+1)}>2\sqrt N\,\eta,
$$

则精确 top-$r$ 一定全部包含在近似 top-$B$ 中。

该证书通常偏保守，因为它使用全局最坏误差，但它是无需分布假设的确定性结论。实验上应该同时报告：

- centered score RMSE；
- top-$B$ 边界 margin；
- 被证书覆盖的 Query 比例；
- 实际 top-k recall，判断理论松弛程度。

### 7.3 由平均 score MSE 限制最多会丢多少核心 token

上一条证书要求所有 token 的误差都足够小。可以用 centered RMSE 得到一个更平滑的保证。

定义精确 top-$r$ 与 top-$B$ 外部的边界：

$$
m_r=s_{(r)}-s_{(B+1)}>0.
$$

再定义：

$$
c_r=
\left\lfloor
\frac{4N\eta^2}{m_r^2}
\right\rfloor,
\qquad
b_r=\min\{r,c_r\}.
$$

则近似 top-$B$ 至少包含 $r-b_r$ 个精确 top-$r$ token。

证明思路如下。令 $e_i=\delta_i-\bar\delta$。满足 $|e_i|\ge m_r/2$ 的 token 数量最多为：

$$
\frac{\sum_i e_i^2}{m_r^2/4}
=
\frac{4N\eta^2}{m_r^2}.
$$

如果一个精确 top-$r$ token 和一个 top-$B$ 外 token 都不属于这些“大误差 token”，二者的相对顺序不会翻转。每丢失一个低误差核心 token，至少要有一个大误差外部 token 占据它的位置。因此总共最多丢失 $b_r$ 个核心 token。

在最坏情况下，被丢失的是核心中 attention 概率最大的 $b_r$ 个 token，所以选中集合的精确 attention mass 至少为：

$$
\sum_{i=b_r+1}^{r}p_{(i)}.
$$

这条结论仍然是确定性的，但比 $2\sqrt N\eta$ 的“一个 token 都不能错”证书更适合实际数据：它允许少量边界 token 交换，同时给出至少保留多少核心 token 和 attention mass 的量化下界。

## 8. 从选中集合到 attention 输出误差

设集合 $S$ 遗漏的完整 attention mass 为：

$$
\epsilon=\sum_{i\notin S}p_i.
$$

把完整 attention 限制到 $S$ 并重新归一化后，有精确恒等式：

$$
\|p-\widetilde p\|_1=2\epsilon.
$$

定义选中与遗漏区域的条件 Value 均值：

$$
o_{\mathrm{in}}=
\frac{1}{1-\epsilon}\sum_{i\in S}p_iv_i,\qquad
o_{\mathrm{out}}=
\frac{1}{\epsilon}\sum_{i\notin S}p_iv_i.
$$

则：

$$
o-\widetilde o
=\epsilon(o_{\mathrm{out}}-o_{\mathrm{in}}),
$$

因此：

$$
\|o-\widetilde o\|_2
\le\epsilon\,\operatorname{diam}\{v_1,\ldots,v_N\}.
$$

这解释了为什么 top-k token recall 不是最关键指标：边界附近可以交换很多低概率 token，只要遗漏的 attention mass 小，attention 输出仍然稳定。

结合上一节，如果精确 top-$r$ 被证书保留，则：

$$
\epsilon\le1-\sum_{i=1}^{r}p_{(i)}.
$$

于是得到完整链条：

$$
\text{QK 量化误差}
\Rightarrow
\text{排序 margin}
\Rightarrow
\text{保留核心 token}
\Rightarrow
\text{遗漏 attention mass}
\Rightarrow
\text{attention 输出误差}.
$$

## 9. 多层误差如何传到最终 logits

设完整第 $\ell$ 层为 $F_\ell$，稀疏第 $\ell$ 层为 $\widetilde F_\ell$。假设在实际到达的状态邻域中：

$$
\|\widetilde F_\ell(x)-\widetilde F_\ell(y)\|_2
\le L_\ell\|x-y\|_2,
$$

并且同一输入上的局部近似误差满足：

$$
\|\widetilde F_\ell(x)-F_\ell(x)\|_2\le a_\ell.
$$

从同一输入开始，递推可得：

$$
\|\widetilde h_L-h_L\|_2
\le
\sum_{\ell=0}^{L-1}
a_\ell\prod_{u=\ell+1}^{L-1}L_u.
$$

若局部误差只来自 attention，则可以进一步使用：

$$
a_\ell
\le
M_\ell
\left(
\sum_h
\epsilon_{\ell,h}^2
\operatorname{diam}(V_{\ell,h})^2
\right)^{1/2},
$$

其中 $\epsilon_{\ell,h}$ 是第 $h$ 个 head 的遗漏 attention mass，$M_\ell$ 是拼接后的多头 attention 输出到该层输出的局部增益。

最终 logits 为 $z=W_Uh_L$。于是：

$$
\|\widetilde z-z\|_\infty
\le
\|W_U\|_{2\rightarrow\infty}
\|\widetilde h_L-h_L\|_2.
$$

把右侧记为 $\rho$，则词表 softmax 还满足：

$$
\operatorname{KL}
(p_{\mathrm{next}}\|\widetilde p_{\mathrm{next}})
\le\frac{\rho^2}{2},
\qquad
\|p_{\mathrm{next}}-\widetilde p_{\mathrm{next}}\|_1
\le\min\{2,\rho\}.
$$

当完整模型 top-1 logit margin 大于 $2\rho$ 时，稀疏模型的 top-1 token 不会改变。这样，理论链不只停留在单层 attention 输出，还给出了最终 next-token 分布和预测稳定性的条件上界。

这个深层界通常会因 Lipschitz 常数乘积而较松。其作用是明确理论闭环和应测量的局部量，不应该被表述为所有输入上的紧保证。

## 10. Query INT8 误差

若：

$$
\widehat q'=q'+e_q,\qquad
\widehat k'=k'-e_k,
$$

则完整 proxy score 误差为：

$$
{q'}^\top k'-{\widehat q'}^\top\widehat k'
={q'}^\top e_k-e_q^\top k'+e_q^\top e_k.
$$

Key allocator 只优化第一项。Query INT8 引入的额外部分必须独立测量，不能全部归因于 Key mixed-bit allocation。

## 11. 复杂度与速度交叉点

完整 attention 每层 decode 的主要复杂度约为：

$$
4H_qNd.
$$

QKSieve 的主要成本为：

$$
2H_qd^2+
c_{\mathrm{idx}}H_qNR+
c_{\mathrm{topk}}H_qN+
4H_qB(N)d.
$$

其中索引扫描仍然是 $O(N)$。加速来自：

- 扫描低比特紧凑 Key 索引，而不是 FP16 Key；
- 最终 AV 只读取 $B(N)$ 个 Value；
- $B(N)$ 在长序列上被 1280 封顶。

设一次性建索引成本为 $I$，Full 和 QKSieve 的稳态单 token 时间分别为 $t_f,t_s$，则生成 $G$ 个 token：

$$
T_f=P+Gt_f,\qquad
T_s=P+I+Gt_s.
$$

若 $t_s<t_f$，请求级 break-even 为：

$$
G^\star=\frac{I}{t_f-t_s}.
$$

这解释了为何 128K 可以明显加速，而 4K/8K 可能仍然更慢。

## 12. 必须补齐的理论验证实验

最终论文至少需要下列图表，且必须按 layer/head 分布而不只报平均值：

| 理论环节 | 需要测量的量 | 目的 |
|---|---|---|
| 精确变换 | $\|AD^\top-I\|_{\max}$、全精度 score 误差 | 验证数值实现没有破坏定理 1 |
| 共同谱 | $\sigma_\ell$、累计 $\sum_{\ell\le r}\sigma_\ell^2$ | 验证共同 QK 信息是否集中 |
| QK 目标 | Key MSE 与 QK MSE 对 recall/mass 的相关性 | 证明 QK weighting 确实更适合检索 |
| PCA 特例 | Query anisotropy 与 QK-balanced 相对 Key-PCA 的收益 | 验证两者何时相同、何时真正不同 |
| band 分解 | cross-band 项占完整 QK MSE 的比例 | 验证可分配近似 |
| bit 分配 | 实际 allocation 与注水公式预测的秩相关 | 验证 Query 敏感度驱动不同层/Head 的 bit |
| Query 漂移 | covariance drift、生成位置、实际 QK MSE | 验证 prompt-tail 校准 |
| held-out regret | $\Omega$、$\Gamma$ 与实际 allocation regret | 验证生成期上界是否有解释力 |
| 排序证书 | centered RMSE、margin、证书覆盖率 | 验证 score 到 top-k 的理论链 |
| attention 质量 | top-k recall、attention mass、输出误差 | 验证 mass 比普通 recall 更关键 |
| 深层传播 | layer output error、final logit KL/top-1/margin | 验证局部误差没有异常放大 |
| 系统模型 | scan/top-k/gather/sparse AV/其他模型时间 | 验证复杂度分析与实测一致 |

## 13. 理论能支持和不能支持的论文结论

可以严格支持：

- 全维 QK-balanced 变换不损失任何 dot product；
- 在独立二阶矩模型下，前 $r$ 个 QK-balanced 方向是期望 score MSE 的全局最优秩 $r$ 子空间；
- QK-MSE 是给定有限样本上的精确 score-MSE；
- 自动 bit 分配在当前 blockwise 目标下是全局最优；
- held-out allocation regret 可由 Query drift 和跨 band 误差共同控制；
- 连续高码率模型给出 Query 敏感度驱动的最优 bit 注水结构；
- 有限 Query 样本误差与真实 decode 分布漂移可以显式分解；
- score 误差范围直接控制 proxy softmax 的 KL 和 $\ell_1$ 偏差；
- Query drift、跨 band 误差、排序稳定性和输出误差都有显式可测上界；
- 遗漏 attention mass 直接控制单层 attention 输出误差。

不能仅靠理论宣称：

- 所有任务都一定保持质量；
- 240 bit 在所有模型上都是最优预算；
- prompt-tail Query 一定代表长生成或多轮 Query；
- 最坏情况下的深层 Lipschitz 上界足够紧；
- 低 score MSE 自动意味着任意 Value 分布下答案不变。

因此正确的论文故事不是“量化误差必然无害”，而是：

> QKSieve 把检索索引的有限 bit 精确分配给 Query 实际敏感的 QK 方向，并建立了从数值近似、排序稳定、attention mass 到模型输出的可审计误差链；实验再验证真实模型通常位于这条误差链的稳定区域。
