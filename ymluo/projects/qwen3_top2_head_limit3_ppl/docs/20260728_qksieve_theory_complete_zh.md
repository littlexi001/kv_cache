# QKSieve 理论分析与完整证明

本文对应当前冻结方法：

- 每层、每个 KV head 独立构造 QK-balanced 坐标；
- 使用 prompt 最后 8 个 Query 位置和 stride-32 的 Key 样本；
- Query 二阶矩 shrinkage 为 0.75；
- 128 维划分为 8 个 16 维 band；
- 每个 band 从 0/1/2/4/8 bit 中选择；
- 每 token、每 KV head 的索引物理预算为 240 bit；
- 扫描完整低比特索引，直接选 top-k；
- 对选中位置使用原始 FP16 K/V 做精确 attention；
- 不使用 rerank、recent/sink 保留、router 或 Full fallback。

文中将“严格恒等式”“统计最优性”和“条件稳定性”分开陈述。最后一类结论需要实验测量其条件，不能被表述为任意任务上的无条件质量保证。

## 1. 数值正定化后的二阶矩

原始经验二阶矩为

$$
\widehat C_k=\frac{1}{m_k}\sum_i k_i k_i^\top,
\qquad
\widehat C_q=\frac{1}{m_q}\sum_j q_j q_j^\top.
$$

先对 Query 做各向同性 shrinkage：

$$
S_q=(1-\lambda)\widehat C_q+
\lambda\frac{\operatorname{tr}(\widehat C_q)}{d}I,
\qquad \lambda=0.75.
$$

对任意半正定矩阵

$$
S=E\operatorname{diag}(\nu)E^\top
$$

定义谱 floor：

$$
\mathcal F(S)=
E\operatorname{diag}\left(\max\{\nu_\ell,\epsilon_S\}\right)E^\top,
$$

其中

$$
\epsilon_S=10^{-8}\max_\ell\nu_\ell+10^{-12}.
$$

实现实际使用

$$
C_k=\mathcal F(\widehat C_k),
\qquad
\widetilde C_q=\mathcal F(S_q).
$$

平方根和逆平方根必须使用同一组 floored eigenvalues。这一点保证后面的双正交恒等式成立。若只在逆平方根中 floor，而平方根仍使用原始特征值，则一般不能严格得到 $AD^\top=I$。

## 2. 双正交变换不引入 full-precision score 误差

令

$$
\widetilde C_q^{1/2}C_k^{1/2}
=U\Sigma V^\top,
$$

并定义

$$
A=\widetilde C_q^{-1/2}U\Sigma^{1/2},
\qquad
D=C_k^{-1/2}V\Sigma^{1/2}.
$$

则

$$
\begin{aligned}
AD^\top
&=
\widetilde C_q^{-1/2}
U\Sigma^{1/2}\Sigma^{1/2}V^\top
C_k^{-1/2}\\
&=
\widetilde C_q^{-1/2}
\left(\widetilde C_q^{1/2}C_k^{1/2}\right)
C_k^{-1/2}\\
&=I.
\end{aligned}
$$

所以对任意 Query 和 Key，不要求它们来自校准数据，

$$
(A^\top q)^\top(D^\top k)
=q^\top AD^\top k
=q^\top k.
$$

因此，完整 128 维 QK-balanced 变换本身不是低秩近似，也不改变精确 QK score。近似只来自后续的低比特索引和 top-k 离散选择。

## 3. Query 和 Key 的二阶矩同时平衡

直接代入可得

$$
\begin{aligned}
A^\top\widetilde C_qA
&=
\Sigma^{1/2}U^\top U\Sigma^{1/2}
=\Sigma,\\
D^\top C_kD
&=
\Sigma^{1/2}V^\top V\Sigma^{1/2}
=\Sigma.
\end{aligned}
$$

因此，变换后的 Query 和 Key 在构造二阶矩下具有相同的对角二阶矩 $\Sigma$。

原始经验矩和构造矩之间仍有可测差异：

$$
\Delta_q^{\mathrm{mom}}
=A^\top(\widehat C_q-\widetilde C_q)A,
$$

$$
\Delta_k^{\mathrm{mom}}
=D^\top(\widehat C_k-C_k)D.
$$

这两个 residual 不影响任意向量上的精确 score 恒等式，但会影响“$\Sigma$ 完全描述实际 Query/Key 能量”的统计解释。因此实验应报告它们，而不能假设为零。

## 4. QK-balanced 是最优 rank-r score 子空间

设相互独立的随机向量 $q,k$ 的非中心二阶矩分别为 $C_q,C_k$。考虑 rank 不超过 $r$ 的双线性近似

$$
q^\top k\approx q^\top B_rk.
$$

令 $E=I-B_r$，则

$$
\begin{aligned}
\mathbb E[(q^\top Ek)^2]
&=
\operatorname{tr}(C_qEC_kE^\top)\\
&=
\left\|
C_q^{1/2}EC_k^{1/2}
\right\|_F^2.
\end{aligned}
$$

所以

$$
\mathbb E[(q^\top k-q^\top B_rk)^2]
=
\left\|
C_q^{1/2}C_k^{1/2}
-C_q^{1/2}B_rC_k^{1/2}
\right\|_F^2.
$$

因为两个二阶矩正定，

$$
B_r\mapsto C_q^{1/2}B_rC_k^{1/2}
$$

在 rank 不超过 $r$ 的矩阵集合上是双射。根据 Eckart-Young-Mirsky 定理，最优近似是

$$
C_q^{1/2}B_r^\star C_k^{1/2}
=U_r\Sigma_rV_r^\top.
$$

因此

$$
B_r^\star
=C_q^{-1/2}U_r\Sigma_rV_r^\top C_k^{-1/2},
$$

最小误差为

$$
\sum_{\ell>r}\sigma_\ell^2.
$$

这说明奇异值顺序对应联合 QK score 能量，而不是单独的 Key reconstruction 能量。

当 $C_q=\alpha I$ 时，QK-balanced 的方向和 Key-PCA 完全相同。因此 Key-PCA 是 Query 各向同性时的特例。

### 4.1 真实 Query--Key 相关性产生的四阶残差

上面的 rank-$r$ 最优性依赖 Query 和 Key 独立抽样。真实 attention 中，同一
位置和同一序列产生的 Query--Key 对可能相关，二阶矩本身不足以描述这种
相关性。定义

$$
H=\mathbb E[(k\otimes q)(k\otimes q)^\top],
\qquad
H_0=C_k\otimes C_q,
\qquad
\Delta_H=H-H_0.
$$

对任意双线性近似 $B$，令

$$
L_{\mathrm{pair}}(B)
=\mathbb E[(q^\top(I-B)k)^2]
$$

表示真实配对分布下的损失，$L_{\mathrm{ind}}(B)$ 表示具有相同二阶矩但
Query/Key 独立抽样时的损失。由

$$
q^\top(I-B)k
=\operatorname{vec}(I-B)^\top(k\otimes q)
$$

可得

$$
\left|L_{\mathrm{pair}}(B)-L_{\mathrm{ind}}(B)\right|
\le
\|\Delta_H\|_{\mathrm{op}}\|I-B\|_F^2.
$$

因此不能把独立模型下的最优性无条件推广到任意真实 attention 配对。
$\Delta_H$ 是缺失的四阶统计量。直接构造完整 $d^2\times d^2$ 矩阵代价较高，
实验可以对冻结候选 $B$ 报告真实 paired loss 与随机 Cartesian loss 的差，
或使用随机投影估计该残差，而不能默认它为零。

## 5. qMSE 目标是精确的有限样本 score MSE

记投影后的 Query 样本为 $q'_j$，某一量化配置产生的 Key error 为 $e_i$。定义

$$
C'_q=\frac{1}{m_q}\sum_jq'_j{q'_j}^\top,
\qquad
C_e=\frac{1}{m_k}\sum_ie_ie_i^\top.
$$

则

$$
\begin{aligned}
\frac{1}{m_qm_k}\sum_{j,i}({q'_j}^\top e_i)^2
&=
\frac{1}{m_qm_k}
\sum_{j,i}
\operatorname{tr}
\left(q'_j{q'_j}^\top e_ie_i^\top\right)\\
&=
\operatorname{tr}(C'_qC_e).
\end{aligned}
$$

这是有限样本恒等式，不需要高斯分布、零均值或 Query/Key 样本独立。

将坐标划分为 band 后，

$$
\operatorname{tr}(C'_qC_e)
=
\sum_g\operatorname{tr}(C'_{q,gg}C_{e,gg})
+
\sum_{g\ne h}\operatorname{tr}(C'_{q,gh}C_{e,hg}).
$$

生产 allocator 最小化第一项。省略的 cross-band 项满足

$$
\left|
\sum_{g\ne h}\operatorname{tr}(C'_{q,gh}C_{e,hg})
\right|
\le
\sum_{g\ne h}
\|C'_{q,gh}\|_F\|C_{e,hg}\|_F.
$$

因此 band 可分性不是隐藏假设，而是可以直接测量的近似误差。

## 6. 240-bit 离散分配的全局最优性

每个 16 维 band 使用

$$
b_g\in\{0,1,2,4,8\}.
$$

只要 band 激活，就额外存一个 FP16 scale。因此归一化 rate 为

$$
r_g(b_g)=b_g+\mathbf 1[b_g>0].
$$

总约束为

$$
\sum_g r_g(b_g)\le15,
$$

等价于

$$
16\sum_g r_g(b_g)\le240\ \mathrm{bit}.
$$

令 $D_g(b)$ 为第 $g$ 个 band 在 bit 数 $b$ 下测得的 qMSE。定义

$$
F(g,r)
$$

为前 $g$ 个 band、总 rate 不超过 $r$ 时的最小 distortion，则

$$
F(g,r)=
\min_{b:r_g(b)\le r}
\left[
F(g-1,r-r_g(b))+D_g(b)
\right].
$$

穷举所有最后一个 band 的选择即可得到最优子结构，因此该动态规划返回完整可行集上的全局最优解。复杂度为

$$
O(GLR'),
$$

其中 $G=8$、$L=5$、$R'=15$。实际 CUDA 代码枚举并缓存同一个有限可行集。

所以在相同坐标、量化器和物理预算下，自动分配的校准 qMSE 不会高于任意固定可行分配。这个结论只针对校准目标；held-out Query 的保证还需要下一节的 drift 条件。

## 7. 为什么 Key-MSE 可能分错 bit

Key-only 目标为

$$
J_K(b)=\operatorname{tr}(C_e(b)),
$$

而 qMSE 目标为

$$
J_{QK}(b)=\operatorname{tr}(C'_qC_e(b)).
$$

若

$$
\mu I\preceq C'_q\preceq LI,
\qquad \mu>0,
$$

令 $b_K$ 和 $b_{QK}$ 分别最小化两个目标，则

$$
\begin{aligned}
J_{QK}(b_K)
&\le L\operatorname{tr}(C_e(b_K))\\
&\le L\operatorname{tr}(C_e(b_{QK}))\\
&\le\frac{L}{\mu}J_{QK}(b_{QK}).
\end{aligned}
$$

即

$$
J_{QK}(b_K)
\le\kappa(C'_q)J_{QK}(b_{QK}).
$$

Query 越各向异性，Key-MSE 的最坏保证越差；当 Query 各向同性时，两者等价。

## 8. 随机正交旋转不能提供有意义的 band 顺序

令 $R$ 为 Haar 随机正交矩阵，$E_g$ 为固定 $d_g$ 维 band 的投影，定义

$$
P_g=RE_gR^\top.
$$

由 Haar 不变性，$\mathbb E[P_g]$ 与任意正交矩阵可交换，因此只能是 $cI$。又因为

$$
\operatorname{tr}(P_g)=d_g,
$$

所以

$$
\mathbb E[P_g]=\frac{d_g}{d}I.
$$

对任意半正定矩阵 $C$，

$$
\mathbb E_R\operatorname{tr}(CP_g)
=\frac{d_g}{d}\operatorname{tr}(C).
$$

随机旋转可以保持完整 dot product，但在期望上所有同维 band 可交换，不能把 QK-sensitive 方向稳定地排到高 bit band。该结论对应 random rotation 消融。

## 9. 校准 Query 到 decode Query 的漂移

对固定 Key error moment $C_e$，令

$$
\Delta=C'_{q,\mathrm{dec}}-C'_{q,\mathrm{cal}}.
$$

因为 $C_e$ 半正定，

$$
|\operatorname{tr}(\Delta C_e)|
\le
\|\Delta\|_{\mathrm{op}}\operatorname{tr}(C_e).
$$

证明方法是对 $C_e$ 做特征分解，并逐个方向使用 Rayleigh quotient 上界。

对任意 allocation $b$，将真实 distortion 写为对角 band 项和 cross-band 项：

$$
J_t(b)=\widetilde J_t(b)+\Gamma_t(b).
$$

若 $b^\star$ 是校准集对角目标的最优解，$\bar b$ 是任意固定可行分配，则

$$
\begin{aligned}
J_{\mathrm{dec}}(b^\star)-J_{\mathrm{dec}}(\bar b)
\le&
\Omega(b^\star)+\Omega(\bar b)\\
&+
|\Gamma_{\mathrm{dec}}(b^\star)|
+|\Gamma_{\mathrm{dec}}(\bar b)|,
\end{aligned}
$$

其中

$$
\Omega(b)=
\sum_g
\|\Delta_{gg}\|_{\mathrm{op}}
\operatorname{tr}(C_{e,gg}(b_g)).
$$

因此，自动 allocation 在 held-out decode Query 上接近最优的充分条件是：

1. 对角 block 的 Query covariance drift 小；
2. cross-band error interaction 小。

这两类量都可以在不使用任务标签的情况下测量。

若校准 Query 独立同分布，令

$$
X_j=q'_j{q'_j}^{\top}-C'_{q,\mathrm{cal}},
$$

并假设几乎处处有

$$
\|X_j\|_{\mathrm{op}}\le L,
\qquad
\|\mathbb E[X_j^2]\|_{\mathrm{op}}\le v.
$$

这里 $L$ 控制单个 Query 对二阶矩的最大扰动，$v$ 是单样本矩阵方差代理。
matrix Bernstein 可将有限样本误差与真实分布漂移分开：

$$
\begin{aligned}
\|\widehat C'_{q,m}-C'_{q,\mathrm{dec}}\|_{\mathrm{op}}
\le&
\sqrt{\frac{2v\log(2d/\delta)}{m}}
+\frac{2L\log(2d/\delta)}{3m}\\
&+
\|C'_{q,\mathrm{cal}}-C'_{q,\mathrm{dec}}\|_{\mathrm{op}}.
\end{aligned}
$$

自回归 Query 通常不是独立同分布，所以该界只能解释有限样本项，不能代替按生成位置和多轮对话测量真实 drift。

### 9.1 少量校准样本对 QK-sensitive 子空间的影响

上一节把变换视为固定。本节进一步分析用少量 prompt-tail Query
估计二阶矩后，变换本身是否稳定。

令总体和经验构造矩分别为

$$
C_q,C_k,\widehat C_q,\widehat C_k\succ0,
$$

并定义

$$
M=C_q^{1/2}C_k^{1/2},
\qquad
\widehat M=\widehat C_q^{1/2}\widehat C_k^{1/2}.
$$

记

$$
\varepsilon_q=\|\widehat C_q-C_q\|_{\mathrm{op}},
\qquad
\varepsilon_k=\|\widehat C_k-C_k\|_{\mathrm{op}},
$$

$$
a_q=
\frac{\varepsilon_q}
{\sqrt{\lambda_{\min}(C_q)}
+\sqrt{\lambda_{\min}(\widehat C_q)}},
\qquad
a_k=
\frac{\varepsilon_k}
{\sqrt{\lambda_{\min}(C_k)}
+\sqrt{\lambda_{\min}(\widehat C_k)}}.
$$

对任意正定矩阵 $X,Y$，平方根扰动满足

$$
\|X^{1/2}-Y^{1/2}\|_{\mathrm{op}}
\le
\frac{\|X-Y\|_{\mathrm{op}}}
{\sqrt{\lambda_{\min}(X)}+\sqrt{\lambda_{\min}(Y)}}.
$$

证明来自 Sylvester 恒等式。令 $Z=X^{1/2}-Y^{1/2}$，则

$$
X^{1/2}Z+ZY^{1/2}=X-Y.
$$

正 Sylvester 算子的逆范数不超过分母的倒数，因而得到上式。
再对乘积加减 $C_q^{1/2}\widehat C_k^{1/2}$，得到

$$
\|\widehat M-M\|_{\mathrm{op}}
\le
\underbrace{
a_q\sqrt{\|\widehat C_k\|_{\mathrm{op}}}
+\sqrt{\|C_q\|_{\mathrm{op}}}\,a_k
}_{\varepsilon_M}.
$$

设总体矩阵 $M$ 在第 $r$ 个方向的奇异值间隔为

$$
\gamma_r=\sigma_r(M)-\sigma_{r+1}(M).
$$

若

$$
\varepsilon_M<\frac{\gamma_r}{2},
$$

则经验和总体左右奇异子空间满足

$$
\max\left\{
\|\sin\Theta(\widehat U_r,U_r)\|_{\mathrm{op}},
\|\sin\Theta(\widehat V_r,V_r)\|_{\mathrm{op}}
\right\}
\le
\frac{2\sqrt{2}\,\varepsilon_M}{\gamma_r}.
$$

证明方法是将 $M$ 写成对称 dilation：

$$
\mathcal H(M)=
\begin{bmatrix}
0&M\\
M^\top&0
\end{bmatrix},
$$

其正特征向量正好对应 $M$ 的左右奇异向量。dilation 的扰动范数仍为
$\|\widehat M-M\|_{\mathrm{op}}$。令 $E=\widehat M-M$，经验左右
奇异子空间相对于 $M$ 的两个残差分别为

$$
R=M\widehat V_r-\widehat U_r\widehat\Sigma_r=-E\widehat V_r,
$$

$$
S=M^\top\widehat U_r-\widehat V_r\widehat\Sigma_r=-E^\top\widehat U_r.
$$

所以 $\|R\|_{\mathrm{op}},\|S\|_{\mathrm{op}}\le\varepsilon_M$。
把两个等式投影到总体奇异子空间的正交补，可得到关于左右
subspace angle 的耦合 Sylvester 系统。Weyl 不等式说明它的谱分离
至少为 $\gamma_r-\varepsilon_M$，从而

$$
\|\sin\Theta\|_{\mathrm{op}}
\le
\frac{\sqrt{\|R\|_{\mathrm{op}}^2+\|S\|_{\mathrm{op}}^2}}
{\gamma_r-\varepsilon_M}
\le
\frac{2\sqrt{2}\,\varepsilon_M}{\gamma_r}.
$$

这个结论说明：

1. shrinkage 和 eigenvalue floor 使分母不为零，控制数值放大；
2. 在 $r=16,32,\ldots,112$ 的 band 边界上，奇异值 gap 决定 band
   是否可由少量样本稳定识别；
3. 若某个边界 gap 很小，方向顺序本身就不稳定，不能仅凭理论声称
   8 个 Query 足够，必须报告样本数消融、子空间夹角和 held-out qMSE；
4. 即使估计方向不准，完整 128 维变换仍满足 $AD^\top=I$。估计误差
   只影响压缩坐标与 bit 分配，不影响未量化 dot product 的严格恒等式。

### 9.2 有限 Query/Key 样本下选择 mixed-bit allocation

上一节控制了二阶矩和 QK-sensitive 子空间的估计误差，但还没有处理一个
离散选择问题：allocator 从有限校准样本上比较许多个 bit allocation，
可能把校准噪声当成真实收益。

先固定 QK-balanced 变换和每个 bit level 的确定性量化器。记物理预算下的
可行 allocation 集合为 $\mathcal B_R$，大小为 $M$。当前设置中

$$
b_g\in\{0,1,2,4,8\},\qquad
\sum_g\left(b_g+\mathbf 1[b_g>0]\right)\le15,
$$

精确枚举得到

$$
M=13{,}817.
$$

对 allocation $b$ 定义 allocator 实际使用的 band-separable loss：

$$
\ell_b(q',k')
=
\sum_g
\left({q'_g}^{\top}e_g(k';b_g)\right)^2.
$$

假设对所有 $b,q',k'$ 都有

$$
0\le \ell_b(q',k')\le L_{\mathrm{loss}}.
$$

令 $m_q$ 个 Query 和 $m_k$ 个 Key 分别独立同分布采样，且两组样本相互
独立。定义总体目标和 Cartesian 校准目标：

$$
J(b)=\mathbb E_{q',k'}[\ell_b(q',k')],
$$

$$
\widehat J(b)
=
\frac{1}{m_qm_k}
\sum_{j=1}^{m_q}\sum_{i=1}^{m_k}
\ell_b(q'_j,k'_i).
$$

虽然这里有 $m_qm_k$ 个配对，但它们复用了 Query 和 Key，不能错误地
当成 $m_qm_k$ 个独立样本。对 Query 和 Key 两层分别使用 Hoeffding
不等式并对 $M$ 个 allocation 做 union bound，可得：以至少
$1-\delta$ 的概率，

$$
\sup_{b\in\mathcal B_R}
|\widehat J(b)-J(b)|
\le
\varepsilon_{\mathrm{sel}},
$$

其中

$$
\varepsilon_{\mathrm{sel}}
=
L_{\mathrm{loss}}
\left[
\sqrt{\frac{\log(4M/\delta)}{2m_q}}
+
\sqrt{\frac{\log(4M/\delta)}{2m_k}}
\right].
$$

若

$$
\widehat b=\arg\min_{b\in\mathcal B_R}\widehat J(b),
\qquad
b^\star=\arg\min_{b\in\mathcal B_R}J(b),
$$

则

$$
J(\widehat b)-J(b^\star)
\le2\varepsilon_{\mathrm{sel}}.
$$

证明只需连续使用三次上面的统一偏差界：

$$
\begin{aligned}
J(\widehat b)
&\le\widehat J(\widehat b)+\varepsilon_{\mathrm{sel}}\\
&\le\widehat J(b^\star)+\varepsilon_{\mathrm{sel}}\\
&\le J(b^\star)+2\varepsilon_{\mathrm{sel}}.
\end{aligned}
$$

若关心包含 cross-band 项的完整 score MSE，令

$$
J_{\mathrm{full}}(b)=J(b)+\Gamma(b)
$$

且 $b_{\mathrm{full}}^\star$ 是其总体最优解，则同样有

$$
J_{\mathrm{full}}(\widehat b)
-J_{\mathrm{full}}(b_{\mathrm{full}}^\star)
\le
2\varepsilon_{\mathrm{sel}}
+|\Gamma(\widehat b)|
+|\Gamma(b_{\mathrm{full}}^\star)|.
$$

这个结果说明：

1. 可行 allocation 数量只通过 $\log M$ 进入界，精确枚举本身不会造成
   指数级统计代价；
2. Cartesian 配对不能把有效样本数夸大为 $m_qm_k$，Query 和 Key
   两侧分别以 $m_q^{-1/2}$ 和 $m_k^{-1/2}$ 收敛；
3. 少量 Query 通常是主要统计瓶颈，stride-32 Key 样本较多时 Key 项更小；
4. 该界要求坐标变换在抽取这些校准样本前已经冻结。生产实现用同一批
   prompt 样本估计变换和 allocation，因此不能直接引用该概率界作为
   生产质量保证。严谨做法是增加独立 split/phase 的 held-out allocation
   audit，并同时报告上一节的子空间扰动量；
5. 最坏界可能很松，所以论文应报告实际的
   $\sup_b|\widehat J(b)-J_{\mathrm{heldout}}(b)|$、selected-allocation
   regret 和 cross-band residual，而不是只报告理论上界。

## 10. 从 score RMSE 到 top-k 保留

令真实和代理 score 分别为 $s_i,\widehat s_i$，并令

$$
\delta_i=s_i-\widehat s_i.
$$

定义误差 range：

$$
R_\delta=\max_i\delta_i-\min_i\delta_i.
$$

若真实 top-k 内 token $i$ 与外部 token $j$ 满足

$$
s_i-s_j>R_\delta,
$$

则

$$
\widehat s_i>\widehat s_j.
$$

因此，远离 top-k 边界的高 margin 核心 token 必然被保留；边界附近近乎并列的 token 允许互换。

令

$$
\bar\delta=\frac{1}{N}\sum_i\delta_i,
\qquad
\eta^2=\frac{1}{N}\sum_i(\delta_i-\bar\delta)^2.
$$

则

$$
R_\delta\le2\sqrt N\,\eta.
$$

若真实排序为

$$
s_{(1)}\ge\cdots\ge s_{(N)}
$$

且某个 $r\le B$ 满足

$$
s_{(r)}-s_{(B+1)}>2\sqrt N\,\eta,
$$

则真实 top-r 全部进入代理 top-B。

更细的 count certificate 为：令

$$
m_r=s_{(r)}-s_{(B+1)},
\qquad
c_r=\left\lfloor\frac{4N\eta^2}{m_r^2}\right\rfloor.
$$

由 Markov 型计数可知，至多 $c_r$ 个 token 的中心化误差绝对值达到 $m_r/2$。因此代理 top-B 至少保留真实 top-r 中的

$$
r-\min\{r,c_r\}
$$

个 token。

还可以得到一个不需要选择 $r$ 的直接 attention-mass 界。令 $T$ 为
真实 top-$B$，$S$ 为代理 top-$B$，并记

$$
\epsilon_T=\sum_{i\notin T}p_i,
\qquad
\epsilon_S=\sum_{i\notin S}p_i.
$$

则

$$
\epsilon_S
\le
\min\left\{1,e^{R_\delta}\epsilon_T\right\}.
$$

证明如下。$T\setminus S$ 和 $S\setminus T$ 数量相同，将两者任意配对。
对每个漏掉的真实 top-$B$ token $i$ 和被选入的替代 token $j$，
由代理 top-$B$ 定义可知

$$
\widehat s_j\ge\widehat s_i.
$$

因此

$$
s_i-s_j
=(\widehat s_i-\widehat s_j)+(\delta_i-\delta_j)
\le R_\delta,
$$

从而

$$
p_i\le e^{R_\delta}p_j.
$$

对所有配对求和，并利用
$p(S\setminus T)\le\epsilon_T$，得到

$$
\begin{aligned}
\epsilon_S
&=\epsilon_T+p(T\setminus S)-p(S\setminus T)\\
&\le
\epsilon_T+(e^{R_\delta}-1)p(S\setminus T)\\
&\le e^{R_\delta}\epsilon_T.
\end{aligned}
$$

结合 $R_\delta\le2\sqrt N\,\eta$，可以把 score RMSE 直接接到
attention mass。这个最坏界会偏松，但比单纯 top-$B$ overlap
更符合模型实际：低概率、近乎并列 token 的互换几乎不影响最终结果。

## 11. 从选中 attention mass 到输出误差

令 $S$ 为选中集合，完整 attention 在集合外的概率质量为

$$
\epsilon=\sum_{i\notin S}p_i.
$$

QKSieve 在 $S$ 内重算精确 score 并重新归一化，所以

$$
\widetilde p_i=
\begin{cases}
p_i/(1-\epsilon), & i\in S,\\
0, & i\notin S.
\end{cases}
$$

直接求和得到精确恒等式

$$
\|p-\widetilde p\|_1=2\epsilon.
$$

定义集合内外的条件 Value 均值：

$$
o_{\mathrm{in}}
=\frac{1}{1-\epsilon}\sum_{i\in S}p_iv_i,
$$

$$
o_{\mathrm{out}}
=\frac{1}{\epsilon}\sum_{i\notin S}p_iv_i.
$$

则

$$
o-\widetilde o
=\epsilon(o_{\mathrm{out}}-o_{\mathrm{in}}),
$$

从而

$$
\|o-\widetilde o\|_2
\le
\epsilon\operatorname{diam}\{v_1,\ldots,v_N\}.
$$

所以 top-k token recall 不是最终理论量。更直接的量是被保留的真实 attention mass，以及集合内外 Value 的差异。

## 12. 层间传播和 next-token 稳定性

设 dense 和 sparse 第 $\ell$ 层映射为 $F_\ell,\widetilde F_\ell$，并假设

$$
\|\widetilde F_\ell(x)-\widetilde F_\ell(y)\|_2
\le L_\ell\|x-y\|_2,
$$

$$
\|\widetilde F_\ell(x)-F_\ell(x)\|_2
\le a_\ell.
$$

令两条执行路径的隐藏状态误差为 $\Delta_\ell$。加减
$\widetilde F_\ell(h_\ell)$ 可得递推：

$$
\Delta_{\ell+1}\le L_\ell\Delta_\ell+a_\ell.
$$

从相同输入开始，

$$
\Delta_L
\le
\sum_{\ell=0}^{L-1}
a_\ell
\prod_{u=\ell+1}^{L-1}L_u.
$$

若 unembedding 后每个 vocabulary logit 的绝对误差不超过 $\rho$，则 softmax 的误差 range 不超过 $2\rho$，从而

$$
\operatorname{KL}
(p_{\mathrm{next}}\|\widetilde p_{\mathrm{next}})
\le\frac{\rho^2}{2},
$$

$$
\|p_{\mathrm{next}}-\widetilde p_{\mathrm{next}}\|_1
\le\min\{2,\rho\}.
$$

当 dense top-1 与第二名的 logit margin 大于 $2\rho$ 时，next-token argmax 不变。

这是条件证书。深层网络的最坏 Lipschitz 常数可能很松，因此实验仍应直接报告 token agreement、KL/PPL 和任务分数。

## 13. Query INT8 误差

若

$$
\widehat q'=q'+e_q,
\qquad
\widehat k'=k'-e_k,
$$

则完整代理 score 误差为

$$
{q'}^\top k'
-{\widehat q'}^\top\widehat k'
=
{q'}^\top e_k
-e_q^\top k'
+e_q^\top e_k.
$$

Key qMSE allocator直接优化第一项。其余 Query 量化项满足

$$
|e_q^\top(k'-e_k)|
\le
\|e_q\|_2\|k'-e_k\|_2.
$$

因此 Query INT8 误差必须独立报告，不能被默认为 Key mixed-bit allocator 已经解释。

## 14. 复杂度、break-even 与速度上界

QKSieve 扫描所有低比特 Key index，所以检索仍为

$$
\Theta(N).
$$

加速来源不是次线性搜索，而是：

1. 扫描每 token、每 KV head 的 30-byte Key index；
2. 将精确 QK 和 AV 限制到

$$
B(N)=\min\{1280,\max(256,\lceil0.06N\rceil)\}.
$$

令 $I_{\mathrm{net}}$ 为没有被 prefill 隐藏的一次性索引成本，$f_t,s_t$ 为 Full 和 QKSieve 在第 $t$ 个 decode step 的同步实测时间。生成 $G$ 个 token 后，QKSieve 更快当且仅当

$$
\sum_{t=1}^{G}(f_t-s_t)>I_{\mathrm{net}}.
$$

若每步时间近似常数，则

$$
G^\star=
\frac{I_{\mathrm{net}}}{t_{\mathrm{full}}-t_{\mathrm{QKSieve}}}.
$$

若在某个短上下文区间始终有

$$
s_t\ge f_t,
$$

则增加输出长度也不能使该区间获得加速；必须降低 scan/top-k/kernel launch 成本，而不是用 Full fallback 掩盖。

对单个 decode step，写成

$$
f=t_{\mathrm{base}}+t_{\mathrm{attn}},
$$

$$
s=t_{\mathrm{base}}+\widetilde t_{\mathrm{attn}},
$$

其中 $\widetilde t_{\mathrm{attn}}$ 已包含 Query 投影、量化、scan、top-k、index append 和 fused gather+exact attention。则

$$
\operatorname{speedup}
\le
\frac{t_{\mathrm{base}}+t_{\mathrm{attn}}}
{t_{\mathrm{base}}}
=
\frac{1}{1-\phi_{\mathrm{attn}}}.
$$

这只是 zero-attention Amdahl 上界，不是可达到的 kernel 性能。

## 15. Query 融合算子的等价性条件

冻结实现先计算

$$
z=qA,
$$

再对每个 16 维 band 计算 scale、INT8 code。设参考路径解量化后的
Query 为 $\widehat z$，融合 kernel 对应结果为
$\widetilde z$；固定 packed Key index 解码后的第 $i$ 个代理 Key
记为 $\widehat k_i$。两条路径的代理 score 差满足

$$
|\widetilde s_i-\widehat s_i|
=
|(\widetilde z-\widehat z)^\top\widehat k_i|
\le
\|\widetilde z-\widehat z\|_2\|\widehat k_i\|_2.
$$

因此令

$$
\epsilon_{\mathrm{fuse}}
=
\|\widetilde z-\widehat z\|_2
\max_i\|\widehat k_i\|_2,
$$

则所有 token 的代理 score 扰动都不超过
$\epsilon_{\mathrm{fuse}}$。若参考路径第 $B$ 与第 $B+1$ 个 score
之间的边界间隔满足

$$
\widehat s_{(B)}-\widehat s_{(B+1)}
>
2\epsilon_{\mathrm{fuse}},
$$

融合路径与参考路径选择完全相同的 top-$B$ 集合。固定相同原始
FP16 K/V 后，两者在实数运算下产生相同的 exact sparse-attention
输出；GPU 上仍需测量有限精度与 reduction 顺序造成的输出误差。

更强地，如果融合 kernel 逐元素产生与参考路径完全相同的 INT8
code 和 FP16 scale，则 packed score 完全相同；除边界 tie 的排列外，
top-k 集合完全相同。这个命题说明融合算子不能只比较 projection
延迟：必须依次核验 code/scale、proxy score、top-k recall、最终
attention output 和完整 selection latency。当前独立 `qfused` 路径
尚未通过 GPU 验证，因此不能替换冻结主方法。

## 16. 理论结论和实验的一一对应

| 理论结论 | 必须测量的实验量 |
|---|---|
| $AD^\top=I$ | 每层每 head 的 $\|AD^\top-I\|_{\max}$ |
| 二阶矩平衡 | 两侧 second-moment residual 和谱 floor residual |
| Query--Key 依赖残差 | 冻结候选映射上的 paired-vs-Cartesian score loss |
| 最优 rank-r score 子空间 | 相同 rank 下 QK-balanced、Key-PCA、random rotation 的 score MSE |
| qMSE identity | 直接 Cartesian score MSE 与 trace 公式数值一致性 |
| band 可分性 | cross-band 项占完整 qMSE 的比例 |
| mixed-bit 最优性 | uniform/fixed/automatic 在相同物理字节下的 held-out qMSE |
| Query covariance 的作用 | QK-balanced+Key-MSE 与完整 qMSE allocation |
| 估计子空间稳定性 | moment operator-norm error、16-D 边界 singular gap、subspace angle |
| Query drift | 1/4/8/16/32 Query、生成位置、1K-4K 输出和多轮对话 |
| 有限 allocation 选择 | 独立 Query/Key split、13,817 个可行解的 uniform calibration-heldout gap、selected regret |
| top-k 保证 | centered score RMSE、boundary margin、oracle/proxy omitted-mass ratio |
| 输出稳定性 | attention-output error、PPL、token agreement、任务分数 |
| Query 融合等价性 | code/scale、score 扰动、边界 margin、top-k recall、最终输出误差 |
| 系统模型 | index build、append、scan、top-k、fused exact attention、whole-model TPOT |

## 17. 可以写进论文和不能写进论文的结论

可以严格写：

- 完整 QK-balanced 变换对任意向量精确保留 dot product；
- 在独立二阶矩模型下，其前 $r$ 维给出最优 rank-r score 近似；
- qMSE 是有限 Query/Key 样本 Cartesian score MSE 的精确 trace 表达；
- 离散 allocator 对可分校准目标是全局最优；
- 在 moment error 小于 band-boundary singular gap 的条件下，估计出的 QK-sensitive 子空间稳定；
- 冻结坐标并使用独立 Query/Key 校准样本时，有限 allocation 选择的总体 regret 至多为两倍 uniform calibration error；
- score margin 和被保留 attention mass 可条件性地控制 attention output；
- proxy top-$B$ 的 omitted mass 不超过 $e^{R_\delta}$ 倍 oracle top-$B$ omitted mass；
- request 是否加速由累计逐 token 节省是否超过一次性索引成本精确决定。

不能无条件写：

- PCA 尾部或低 bit 信息对所有任务都无关；
- 240-bit index 必然保留任意 Query 的 top-k；
- prompt 最后 8 个 Query 必然代表长输出或多轮对话；
- 小 attention-output error 在任意深层网络中必然得到相同答案；
- 1% active attention token 等价于只存 1% KV memory。

当前方法保留完整 FP16 K/V，并额外存储约 5.859% 的 Key 检索索引。它的定位应是 GPU-resident exact-KV retrieval accelerator，而不是 KV memory compression 或 offload 方法。
