# RoPE 对长程检索的影响：第一层相位、跨层累计与改进方案

> 模型：Qwen3-8B；理论与文献核对日期：2026-07-30。  
> 核心问题：为什么只在证据与问题之间增加少量文本，就可能使正确证据的 QK、attention mass 和最终答案概率大幅下降？

## 阅读结构

本文严格按照四个研究问题组织：

1. **第一层：**正确证据的 RoPE QK 如何随相对位置相位变化；
2. **第 $N$ 层到第 $N+1$ 层：**第一层扰动如何经 attention、Value、
   residual 和 MLP 逐层累计；
3. **近期工作：**已有长上下文 RoPE 方法解决了什么，还缺少什么；
4. **Final：**设计面向远程证据检索的 SAGE-RoPE，并与 exact
   post-RoPE Top-2% 比较。

公式推导和方案设计位于前四节；GPU7 实验支撑、限制和复现路径放在后面。

## 结论先行

RoPE 对远程证据的影响不是简单的“距离越远，分数必然越低”，而是：

1. 第一层中，增加文本会改变 Query 与固定证据 Key 的相对相位；每个二维频率对的贡献随距离周期性增减。
2. 第一层产生的小扰动会先改变 softmax 权重和 Value 写入，再经过后续层的 Q/K/V、attention、MLP 和残差连接继续传播。
3. 因而，后层看到的已经不只是“同一个 Query 加了不同 RoPE”，而是内容向量本身也变了，即 pre-RoPE Query 漂移。
4. 长程检索需要将“近程顺序”与“远程语义相关性”拆开处理。直接把所有位置编码关闭，或完全采用 pre-RoPE 分数，都不稳定。
5. 在 24 个全新 seeds 上，SAGE dual-max 在 32K 将 Gold PPL 从
   8.056 降至 5.605、首 token 准确率从 20.8% 提高至 37.5%，且
   $\Delta$NLL 的 95% bootstrap 区间不跨 0；但 64K 只有改善趋势，
   8K–16K 没有 PPL 优势。因此它是长程补偿原型，不应无条件替代
   exact post-RoPE Top-2%。

本文提出的候选方案是 **SAGE-RoPE：Semantic-Adaptive Gated Evidence RoPE**：

- 近程 token：保留标准 RoPE；
- 远程 token：用 pre-RoPE 语义分数召回；
- 每层每个 head 校准语义分数尺度；
- 将校准后的语义分数与原 post-RoPE 分数连续融合，而不是硬切换；
- 只让少量远程候选进入 softmax，避免长尾分母重新淹没证据；
- 检索到证据块后，以块为单位进行位置修复，保留块内顺序。

---

## 1. 第一层：RoPE 怎样改变 QK 分数

### 1.1 相对距离与频率

设 Query 位于位置 $t$，正确证据 Key 位于位置 $p$。定义 Query 相对证据的距离：

$$
\Delta=t-p.
$$

其中：

- $t$：Query 所在位置；
- $p$：证据 Key 所在位置；
- $\Delta$：Query 相对证据的距离。

例如，证据在第 100 个 token，问题 token 在第 1000 个 token，则：

$$
\Delta=1000-100=900.
$$

RoPE 将一个 head 的 hidden dimension 两两组成二维平面。第 $i$ 个二维平面的旋转角速度为：

$$
\omega_i
=
\theta^{-2i/d},
\qquad
i=0,\ldots,\frac d2-1.
$$

Qwen3-8B 中：

$$
d=128,
\qquad
\theta=10^6,
\qquad
i=0,\ldots,63.
$$

因此，一个 head 的 128 个维度被组成 64 个二维旋转对：

- 小 $i$ 时，$\omega_i$ 较大，旋转快，属于高频；
- 大 $i$ 时，$\omega_i$ 较小，旋转慢，属于低频；
- 相对距离增加 $\Delta$ 后，第 $i$ 对累积的位置相位为 $\Delta\omega_i$。

这里的 $\omega_i=\theta^{-2i/d}$ 是标准 RoPE 的基础频率。若推理时使用 YaRN 等缩放方法，还需要将它替换为模型实际使用的有效频率；第 1.6 节会给出此前边界实验中的有效周期。

### 1.2 为什么只剩“相对位置”

记未旋转的 Query 和 Key 为 $q,k$，位置 $x$ 对应的分块旋转矩阵为 $R_x$。标准 RoPE QK 分数为：

$$
s(t,p)
=
\frac{1}{\sqrt d}
\left(R_tq\right)^\top
\left(R_pk\right).
$$

因为旋转矩阵满足：

$$
R_t^\top=R_{-t},
\qquad
R_aR_b=R_{a+b},
$$

所以：

$$
R_t^\top R_p
=
R_{-t}R_p
=
R_{p-t}.
$$

代回 QK 分数：

$$
\begin{aligned}
s(t,p)
&=
\frac{1}{\sqrt d}
q^\top R_t^\top R_pk\\
&=
\frac{1}{\sqrt d}
q^\top R_{p-t}k\\
&=
\frac{1}{\sqrt d}
q^\top R_{-\Delta}k.
\end{aligned}
$$

因此，两个绝对位置 $t,p$ 被合并成相对差：

$$
p-t=-\Delta.
$$

这就是 RoPE 的关键性质：注意力并不直接记住“第 100 位”和“第 1000 位”，而是主要通过二者相差 900 个 token 所对应的相对旋转关系工作。

### 1.3 单个二维频率对的精确展开

取第 $i$ 个二维对：

$$
q_i=
\begin{bmatrix}
q_{i,x}\\
q_{i,y}
\end{bmatrix},
\qquad
k_i=
\begin{bmatrix}
k_{i,x}\\
k_{i,y}
\end{bmatrix}.
$$

令：

$$
\alpha_i=\Delta\omega_i.
$$

按照本文采用的旋转矩阵约定：

$$
R(-\alpha_i)
=
\begin{bmatrix}
\cos\alpha_i & \sin\alpha_i\\
-\sin\alpha_i & \cos\alpha_i
\end{bmatrix}.
$$

这一二维对未经 $1/\sqrt d$ 缩放的 QK 贡献为：

$$
s_i(\Delta)
=
q_i^\top R(-\alpha_i)k_i.
$$

直接展开：

$$
\begin{aligned}
s_i(\Delta)
&=
q_{i,x}
\left(
k_{i,x}\cos\alpha_i+k_{i,y}\sin\alpha_i
\right)\\
&\quad+
q_{i,y}
\left(
-k_{i,x}\sin\alpha_i+k_{i,y}\cos\alpha_i
\right)\\
&=
\left(q_{i,x}k_{i,x}+q_{i,y}k_{i,y}\right)\cos\alpha_i\\
&\quad+
\left(q_{i,x}k_{i,y}-q_{i,y}k_{i,x}\right)\sin\alpha_i.
\end{aligned}
$$

定义：

$$
A_i
=
q_{i,x}k_{i,x}
+
q_{i,y}k_{i,y},
$$

$$
B_i
=
q_{i,x}k_{i,y}
-
q_{i,y}k_{i,x}.
$$

其中：

- $A_i$ 是普通二维点积，表示 Query 与 Key 在该二维平面上的同方向相似性；
- $B_i$ 是二维叉积的有向版本，表示二者在该平面上的方向错位。

于是：

$$
s_i(\Delta)
=
A_i\cos(\Delta\omega_i)
+
B_i\sin(\Delta\omega_i).
$$

完整的一个 head 的 QK logit 是 64 个二维对贡献之和，再乘以缩放因子：

$$
s(t,p)
=
\frac{1}{\sqrt d}
\sum_{i=0}^{d/2-1}
s_i(\Delta).
$$

因此，一个远程证据的 QK 分数并不是由单一的“距离衰减函数”决定，而是由 64 个不同频率、不同内容系数的振荡项叠加得到。

### 1.4 改写成“振幅 + 相位”

定义：

$$
\rho_i
=
\sqrt{A_i^2+B_i^2},
\qquad
\psi_i
=
\operatorname{atan2}(B_i,A_i).
$$

这相当于将二维向量 $(A_i,B_i)$ 写成极坐标：

$$
A_i=\rho_i\cos\psi_i,
\qquad
B_i=\rho_i\sin\psi_i.
$$

代入上一节：

$$
\begin{aligned}
s_i(\Delta)
&=
\rho_i\cos\psi_i\cos(\Delta\omega_i)
+
\rho_i\sin\psi_i\sin(\Delta\omega_i)\\
&=
\rho_i\cos\left(\Delta\omega_i-\psi_i\right).
\end{aligned}
$$

即：

$$
s_i(\Delta)
=
\rho_i
\cos\left(\Delta\omega_i-\psi_i\right).
$$

这个形式可以直接解释每个量的作用：

- $\rho_i$：该二维对最多能够贡献多大的 QK 分数；
- $\psi_i$：当前 Query 与 Key 的内容向量本身偏好的相位；
- $\Delta\omega_i$：相对位置带来的相位；
- $\Delta\omega_i-\psi_i$：内容偏好与位置旋转之间的失配量。

当二者接近时：

$$
\Delta\omega_i-\psi_i\approx 0,
$$

余弦接近 1，该频率对产生较大的正贡献。当相差接近 $\pi$ 时：

$$
\Delta\omega_i-\psi_i\approx\pi,
$$

余弦接近 $-1$，即使 Key 的内容确实是正确证据，该频率对也可能产生较大的负贡献。

所以，“证据语义正确”并不等于“RoPE 后它必然得到正向加成”。内容决定 $\rho_i,\psi_i$，位置决定 $\Delta\omega_i$，最终 QK 是两者共同作用的结果。

### 1.5 插入 $m$ 个 filler token 后发生什么

保持证据位置不变，在证据与 Query 之间插入 $m$ 个 filler token，使 Query 向后移动：

$$
\Delta'=\Delta+m.
$$

第 $i$ 个二维对的分数变化为：

$$
\delta s_i
=
s_i(\Delta+m)-s_i(\Delta).
$$

直接代入：

$$
\begin{aligned}
\delta s_i
&=
A_i
\left[
\cos\left((\Delta+m)\omega_i\right)
-
\cos(\Delta\omega_i)
\right]\\
&\quad+
B_i
\left[
\sin\left((\Delta+m)\omega_i\right)
-
\sin(\Delta\omega_i)
\right].
\end{aligned}
$$

使用三角恒等式：

$$
\cos x-\cos y
=
-2\sin\left(\frac{x+y}{2}\right)
\sin\left(\frac{x-y}{2}\right),
$$

$$
\sin x-\sin y
=
2\cos\left(\frac{x+y}{2}\right)
\sin\left(\frac{x-y}{2}\right),
$$

得到精确变化式：

$$
\delta s_i
=
2\sin\left(\frac{m\omega_i}{2}\right)
\left[
-A_i
\sin\left(\frac{(2\Delta+m)\omega_i}{2}\right)
+
B_i
\cos\left(\frac{(2\Delta+m)\omega_i}{2}\right)
\right].
$$

这个式子包含两部分。外部因子

$$
\sin\left(\frac{m\omega_i}{2}\right)
$$

表示插入 $m$ 个 token 后产生了多大的相位增量；方括号中的内容—相位耦合项，则决定在当前 $A_i,B_i$ 和当前相位下，这次变化最终是增益还是损失。

因此，分数并不会随距离单调下降，而是可能下降、恢复甚至反复穿过零点。

因为：

$$
\left|
-A_i\sin x+B_i\cos x
\right|
\le
\sqrt{A_i^2+B_i^2}
=
\rho_i,
$$

所以有：

$$
\left|\delta s_i\right|
\le
2\rho_i
\left|
\sin\left(\frac{m\omega_i}{2}\right)
\right|.
$$

对于完整的一个 head，插入 filler 前后的 QK logit 变化为：

$$
\delta s_{\mathrm{head}}
=
\frac{1}{\sqrt d}
\sum_{i=0}^{d/2-1}
\delta s_i.
$$

利用三角不等式，可以得到一个保守上界：

$$
\left|
\delta s_{\mathrm{head}}
\right|
\le
\frac{2}{\sqrt d}
\sum_{i=0}^{d/2-1}
\rho_i
\left|
\sin\left(\frac{m\omega_i}{2}\right)
\right|.
$$

但实际变化通常小于这个上界，因为不同频率对的 $\delta s_i$ 可能一部分为正、一部分为负，彼此抵消。只有当若干个高能量频率对在同一长度区间内共同下降时，完整 head 的证据 QK 才会出现明显下跌。

该上界给出三个重要结论：

1. 高频对的 $\omega_i$ 较大，少量新增 token 就可能造成显著相位变化；
2. 但只有当 $\rho_i$ 较大时，相位变化才会转化成显著 QK 变化；
3. 因此，不能只根据频率高低判断一个 head 负责局部信息还是远程信息，还必须观察该 head 在具体 Q/K 上学到的内容能量和相位。

对距离求导，可以得到当前位置的局部敏感度：

$$
\frac{\partial s_i}{\partial\Delta}
=
\omega_i
\left[
-A_i\sin(\Delta\omega_i)
+
B_i\cos(\Delta\omega_i)
\right].
$$

它表示：在当前位置继续增加少量距离时，该二维对的 QK 分数会变化多快。局部敏感度同时受到三项控制：

- 频率 $\omega_i$；
- 当前距离对应的相位 $\Delta\omega_i$；
- 内容系数 $A_i,B_i$。

这也解释了为什么同样增加 8K token，不同 head、不同证据甚至相邻长度点的 QK 变化都可能完全不同。

### 1.6 Qwen3-8B 的数值尺度

Qwen3-8B 原始 RoPE base 为 $10^6$。在此前 140K 边界实验中，实际使用了 YaRN factor 4；以下是模型实际生成的部分有效频率：

| 二维对 $i$ | 有效周期 | 增加 64 token 的相位变化 | 距离 143,424 的相位 |
|---:|---:|---:|---:|
| 0 | 6.28 | 64.000 rad | 143,424.0 rad |
| 8 | 35.33 | 11.381 rad | 25,504.8 rad |
| 16 | 198.69 | 2.024 rad | 4,535.5 rad |
| 24 | 1,117.33 | 0.360 rad | 806.5 rad |
| 32 | 9,518.09 | 0.042 rad | 94.7 rad |
| 40 | 110,325.34 | 0.00364 rad | 8.17 rad |
| 48 | 715,290.38 | 0.00056 rad | 1.26 rad |
| 63 | 18,227,720.0 | 0.000022 rad | 0.049 rad |

因此，只增加 64 个 token：

- 对前若干高频和中频对已是明显旋转；
- 对低频对直接相位变化很小；
- 但第一层 attention 输出的改变会进入下一层，之后不再只是直接相位变化。

### 1.7 GPU7 上的第一层精确验证

在 Qwen3-8B、YaRN factor 2、NF4、seed 0 上，固定同一条证据和同一
Query token，只改变两者距离。第一层的 pre-RoPE Q 和 K 在四个长度下
逐位完全相等：

$$
\widetilde Q_0(\Delta_1)
=
\widetilde Q_0(\Delta_2),
\qquad
\widetilde K_0(\Delta_1)
=
\widetilde K_0(\Delta_2).
$$

它们的范数始终分别为 88.2608 和 915.2758。因此下面的第一层 QK
变化不可能来自内容向量漂移，只能来自 $\Delta\omega_i$ 的变化：

| 证据—Query 距离 | 32 个 head 的平均证据 QK | 最小 QK | 最大 QK | 公式重构最大误差 |
|---:|---:|---:|---:|---:|
| 7,996 | -5.5876 | -15.6637 | 3.3864 | 0.00063 |
| 16,188 | -4.8579 | -14.2073 | 1.1108 | 0.00109 |
| 32,572 | -13.4190 | -26.6061 | 1.8319 | 0.00131 |
| 65,340 | -33.5855 | -66.4306 | 1.4269 | 0.00198 |

把 64 个二维对的
$A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i)$ 相加后，与显式旋转
Q/K 得到的分数最大只相差 0.00198，数值上验证了上述推导。

![第一层 RoPE 频带贡献](../artifacts/20260730_first_layer_rope_phase_gpu7/analysis/first_layer_rope_phase_bands.png)

图中的四条频带线是为了可读性将 64 个二维对按频率平均分为四组，
不是说 Qwen3-8B 只有四个频率。这个样例中，高、中频贡献发生振荡，
最低频组在长距离处形成主要负贡献。第一层平均 QK 为负也不等于模型
已经回答失败：许多真正承担证据读取的 head 位于后层；该实验的作用是
严格隔离并验证第一步“纯相位扰动”。

---

## 2. 从第 $N$ 层到第 $N+1$ 层：扰动如何累计

先给出结论：**固定参数意味着存在从完整输入到最终概率的精确函数，
但只知道最终 Query 那一行的 $\delta h_0$ 并不够。** 应当把第 0 层
完整状态写成：

$$
X_0=(H_0,P,M),
$$

其中 $H_0$ 包含全部 token 的 hidden states，$P$ 是位置，$M$ 是
causal mask。增加 filler 后，不仅 Query—证据距离改变，还增加了新的
K/V 和 softmax 竞争项。在本实验中，同一个 Query token 的
$\delta h_0=0$，模型输出仍然发生翻转；直接扰动是通过完整上下文进入
每一层 attention 的。

若把第 $l$ 层完整计算记为 $\Phi_l$，最终 Query 行抽取算子记为
$\pi_q$，则精确的“超级公式”确实存在：

$$
z(X_0)
=
W_U\mathcal N_f
\left(
\pi_q
\left[
\Phi_{L-1}\circ\cdots\circ\Phi_0(X_0)
\right]
\right),
$$

$$
p_g(X_0)
=
\frac{\exp(z_g(X_0))}
{\sum_v\exp(z_v(X_0))}.
$$

这里 $\mathcal N_f$ 是最终 RMSNorm，$W_U$ 是输出矩阵。该式精确，
但计算它仍然等价于执行一次前向；它不是仅凭一个差异范数就能跳过
attention、softmax、MLP 的闭式捷径。

### 2.1 精确的层更新

以 pre-norm Transformer 为例，第 $l$ 层可写为：

$$
u_l
=
h_l
+
\mathcal A_l(h_l;P),
$$

$$
h_{l+1}
=
u_l
+
\mathcal M_l(u_l),
$$

其中：

- $h_l$ 是进入该层的 residual；
- $\mathcal A_l$ 是 attention 分支；
- $\mathcal M_l$ 是 MLP 分支；
- $P$ 表示位置及 RoPE 相位。

比较短序列 $S$ 与增加 filler 后的长序列 $T$，定义：

$$
\delta h_l
=
h_{l,T}-h_{l,S}.
$$

则精确差分为：

$$
\delta h_{l+1}
=
\delta h_l
+
\delta\mathcal A_l
+
\delta\mathcal M_l.
$$

这就是此前实验中“residual 输入差异 + attention 差异 + MLP 差异 = residual 输出差异”的来源。

上式是在实数运算下的恒等式。Qwen3-8B 实验实际使用 BF16，两个
residual add 会分别舍入。若将 BF16 舍入算子记为 $Q_b$，硬件实际
执行的精确递推是：

$$
u_{l,c}
=
Q_b\!\left(h_{l,c}+\mathcal A_{l,c}\right),
$$

$$
h_{l+1,c}
=
Q_b\!\left(u_{l,c}+\mathcal M_{l,c}\right),
\qquad c\in\{S,T\},
$$

因此：

$$
\delta u_l
=
Q_b\!\left(h_{l,T}+\mathcal A_{l,T}\right)
-
Q_b\!\left(h_{l,S}+\mathcal A_{l,S}\right),
$$

$$
\delta h_{l+1}
=
Q_b\!\left(u_{l,T}+\mathcal M_{l,T}\right)
-
Q_b\!\left(u_{l,S}+\mathcal M_{l,S}\right).
$$

这组式子既包含非线性的 attention、softmax、RMSNorm、SiLU，也包含
实际数值精度，因此可以逐层精确重放。

### 2.2 QK 分数的三类扰动

第 $l$ 层、证据位置 $j$ 的分数为：

$$
s_{l,j}
=
\frac{
q_l^\top
R_{\Delta_j}
k_{l,j}
}{
\sqrt d
}.
$$

一阶展开：

$$
\delta s_{l,j}
\approx
\frac{1}{\sqrt d}
\left[
\delta q_l^\top R_{\Delta_j}k_{l,j}
+
q_l^\top R_{\Delta_j}\delta k_{l,j}
+
q_l^\top\delta R_{\Delta_j}k_{l,j}
\right].
$$

三项分别代表：

1. Query 内容漂移；
2. Key 内容漂移；
3. 相对位置相位变化。

第一层最终 Query token 的输入 embedding 相同，因此最初的 $\delta q_0$ 可以为零；但第三项先改变 attention。到后续层，前一层 Value 写入不同，第一项和第二项随即变为非零。

### 2.3 QK 如何变成 attention mass 变化

attention 权重为：

$$
a_{l,j}
=
\frac{\exp(s_{l,j})}
{\sum_r\exp(s_{l,r})}.
$$

softmax 的 Jacobian 为：

$$
\frac{\partial a_{l,j}}{\partial s_{l,k}}
=
a_{l,j}
\left(
\mathbf 1[j=k]-a_{l,k}
\right).
$$

因此，正确证据 $g$ 的一阶变化为：

$$
\delta a_{l,g}
\approx
a_{l,g}
\left(
\delta s_{l,g}
-
\sum_j a_{l,j}\delta s_{l,j}
\right).
$$

也就是说，关键不只是证据分数是否下降，而是：

$$
\delta s_{l,g}
<
\sum_j a_{l,j}\delta s_{l,j}
$$

时，证据相对于竞争者的 attention mass 才会下降。

### 2.4 attention mass 如何改变下一层 Query

attention 输出为：

$$
o_l
=
W_{O,l}
\sum_j a_{l,j}v_{l,j}.
$$

其一阶变化为：

$$
\delta o_l
\approx
W_{O,l}
\left[
\sum_j\delta a_{l,j}v_{l,j}
+
\sum_j a_{l,j}\delta v_{l,j}
\right].
$$

第一项表示“写入哪些 token 变了”，第二项表示“被写入的 Value 本身变了”。

随后：

$$
\delta h_{l+1}
\approx
\delta h_l
+
\delta o_l
+
J_{\mathcal M_l}
\left(
\delta h_l+\delta o_l
\right).
$$

下一层的 pre-RoPE Query 因而变成：

$$
\delta q_{l+1}^{\mathrm{pre}}
\approx
J_{Q,l+1}\delta h_{l+1}.
$$

这就是从“第一层只发生相位变化”逐步变成“后层 pre-RoPE Query 内容漂移”的完整链条。

### 2.5 多层累计形式

上面的有限差分递推是精确的，但它需要同时计算短、长两条非线性轨迹。
若希望只在短序列轨迹附近预测小扰动，才使用一阶 Jacobian。将一层
对 Query 状态的局部 Jacobian 记为 $J_l$，该层由其他 token、位置和
mask 直接注入的扰动记为 $b_l$，则：

$$
\delta h_{l+1}
\approx
J_l\delta h_l+b_l.
$$

递推到第 $L$ 层：

$$
\delta h_L
\approx

\left(
\prod_{r=0}^{L-1}J_r
\right)
\delta h_0
+
\sum_{l=0}^{L-1}
\left(
\prod_{r=l+1}^{L-1}J_r
\right)b_l.
$$

因此，小扰动是否被放大，取决于后续 Jacobian 乘积在该扰动方向上的增益，而不只取决于单层 RoPE 的旋转幅度。

对本实验的最终 Query，$\delta h_0=0$，所以变化全部来自各层的
$b_l$；如果只保留第一项，错误地从零出发，就会预测“最终没有变化”。

还要注意：模型在最后一层 residual 后存在 final RMSNorm。定义：

$$
r_L=\mathcal N_f(h_L),
$$

则正确答案与竞争答案的精确 margin 为：

$$
\Delta_{\mathrm{out}}
=
\left(
W_{\mathrm{gold}}
-
W_{\mathrm{comp}}
\right)^\top r_L.
$$

其变化为：

$$
\delta\Delta_{\mathrm{out}}
=
\left(
W_{\mathrm{gold}}
-
W_{\mathrm{comp}}
\right)^\top
\left[
\mathcal N_f(h_{L,T})
-
\mathcal N_f(h_{L,S})
\right].
$$

当该量使 margin 穿过 0，首 token 的赢家就会切换。

精确有限差分与一阶传播必须区分：

- 精确式使用真实的 $\delta\mathcal A_l$、$\delta\mathcal M_l$ 和
  final RMSNorm 差分，可以一直重构到最终概率；
- Jacobian/JVP 式只在扰动足够小时近似成立，其误差可能随层数累计；
- 概率还依赖全词表 softmax，不能仅由 gold 对单个 competitor 的
  margin 唯一决定。

### 2.6 143,424 → 143,488 的逐层实验验证

实验使用 Qwen3-8B、BF16、YaRN factor 4；两条输入仅相差末端新增的
64 个 filler token，并比较同一个最终 Query token。

| 层 | residual 输出绝对差异 | pre-RoPE Q 相对差异 | Q cosine |
|---:|---:|---:|---:|
| L0 | 1.466 | 0.000 | 1.000 |
| L19 | 11.574 | 0.103 | 0.995 |
| L20 | 15.976 | 0.120 | 0.993 |
| L21 | 23.128 | 0.190 | 0.982 |
| L22 | 35.106 | 0.217 | 0.976 |
| L23 | 47.462 | 0.340 | 0.941 |
| L24 | 59.158 | 0.350 | 0.937 |

表中比较的是 143,424-token 条件 $S$ 和 143,488-token 条件 $T$
下同一个最终 Query token。三个量分别定义为：

$$
D_{\mathrm{res},l}
=
\left\|
h_{l,T}^{\mathrm{out}}
-
h_{l,S}^{\mathrm{out}}
\right\|_2,
$$

$$
D_{Q,l}^{\mathrm{rel}}
=
\frac{
\left\|
\widetilde Q_{l,T}
-
\widetilde Q_{l,S}
\right\|_F
}{
\left\|
\widetilde Q_{l,S}
\right\|_F
},
$$

$$
C_{Q,l}
=
\frac{
\left\langle
\widetilde Q_{l,T},
\widetilde Q_{l,S}
\right\rangle
}{
\left\|\widetilde Q_{l,T}\right\|_F
\left\|\widetilde Q_{l,S}\right\|_F
}.
$$

其中 residual 是 4096 维层输出；$\widetilde Q_l$ 是 RoPE 前
$32\times128$ 的全部 Query heads。Residual 一列是绝对 L2 距离，
而 pre-RoPE Q 一列除以了短序列 Q 的范数；所以 `0.103` 表示扰动范数
约为原 Q 范数的 10.3%，`11.574` 则不是百分比。

该表与推导一致：

- L0 的 pre-RoPE Q 完全相同；
- L0 residual 输出已不同，说明直接 RoPE/attention 扰动已经写入；
- 差异随后逐层累计；
- 到 L20–L24，pre-RoPE Q 本身发生显著漂移。

进一步使用两条前向中实际记录的 attention、MLP 和 residual 向量，
逐层重构 $N\rightarrow N+1$：

必须先做**向量相加**，再取范数。为避免把三个范数误认为可以直接
相加，下表将三项范数压在同一列，并明确给出重构向量的范数：

| 层 | 三项范数 $\|\delta h_l\|/\|\delta\mathcal A_l\|/\|\delta\mathcal M_l\|$ | 实数向量和 $\|\delta h_l+\delta\mathcal A_l+\delta\mathcal M_l\|$ | BF16 精确重构范数 | 实际 $\|\delta h_{l+1}\|$ | BF16 最大误差 |
|---:|---:|---:|---:|---:|---:|
| L0 | 0.000 / 0.542 / 1.293 | 1.464 | 1.466 | 1.466 | 0 |
| L8 | 4.615 / 1.548 / 3.205 | 5.102 | 5.106 | 5.106 | 0 |
| L16 | 7.605 / 3.736 / 5.199 | 8.233 | 8.276 | 8.276 | 0 |
| L20 | 11.574 / 8.642 / 10.103 | 15.964 | 15.976 | 15.976 | 0 |
| L21 | 15.976 / 11.379 / 14.070 | 23.117 | 23.128 | 23.128 | 0 |
| L22 | 23.128 / 20.612 / 22.013 | 35.112 | 35.106 | 35.106 | 0 |
| L23 | 35.106 / 21.171 / 26.279 | 47.468 | 47.462 | 47.462 | 0 |
| L24 | 47.462 / 25.275 / 26.395 | 59.156 | 59.158 | 59.158 | 0 |
| L28 | 87.939 / 27.650 / 42.217 | 102.442 | 102.604 | 102.604 | 0 |
| L35 | 269.676 / 59.114 / 130.886 | 313.938 | 312.676 | 312.676 | 0 |

三个范数之和不应等于输出范数，因为：

$$
\begin{aligned}
\|x+y+z\|_2^2
={}&\|x\|_2^2+\|y\|_2^2+\|z\|_2^2\\
&+2\langle x,y\rangle
+2\langle x,z\rangle
+2\langle y,z\rangle.
\end{aligned}
$$

只有三个向量完全同向时，输出范数才等于三个范数之和。以 L0 为例，
$\delta h_0=0$，attention 与 MLP 差分向量的夹角为 $82.72^\circ$；
所以 $0.542+1.293=1.835$ 并不是公式预测，正确的向量计算是
$\|\delta\mathcal A_0+\delta\mathcal M_0\|_2=1.464$。再计入两次
BF16 舍入后得到 1.466，与实际输出完全一致。

这里 L0 的 $\|\delta h_0\|_2=0$，但 attention 分支已经出现
$0.542$ 的差异，再次说明只知道 Query 的第 0 层输入变化不够。36 层
相邻的 `residual_out → residual_in` 最大误差也是 0。若直接使用实数
恒等式 $\delta h_{l+1}=\delta h_l+\delta\mathcal A_l+
\delta\mathcal M_l$ 而忽略 BF16 两次舍入，单层最大相对误差为
5.95%；加入 $Q_b$ 后，36 层重构误差均为 0。

最后把 L35 residual 送入真实 final RMSNorm 和输出矩阵，结果为：

| 指标 | 143,424 | 143,488 实际值 | 由差分公式重构 | 重构误差 |
|---|---:|---:|---:|---:|
| BF16 `P(nine)` | 42.620% | 1.745% | — | 回放误差 $<1.9\times10^{-5}$ |
| BF16 `nine` margin | +1.000 | -2.750 | — | 0 |
| FP32 `P(nine)` | 41.688% | 1.801227% | 1.801231% | $3.73\times10^{-8}$ |
| FP32 `nine` margin | +1.001644 | -2.752247 | -2.752243 | $3.81\times10^{-6}$ |
| FP32 全词表 logit | — | — | $z_S+W_U\delta r_L$ | 最大误差 $7.63\times10^{-6}$ |

这验证了：只要保留完整状态和每层非线性有限差分，公式确实能够一直
重构到最终概率和 margin。它并不意味着只用一个 $\delta h_0$ 标量，
或只计算一次矩阵乘法，就能预测长模型的输出。

作为近似误差对照，只把最终 RMSNorm 在短序列点作一阶 JVP 时，
归一化向量差异的相对误差已经达到 9.86%，margin 变化预测为
-3.4547，而精确值是 -3.7539，误差 0.2992。完整 36 层 Jacobian 乘积
还会继续累积各层线性化误差，所以理论推导应把它标为局部近似，而不是
精确等号。

最后，已有因果 patch 提供了非恒等式证据：保持 143,488 的长上下文
不变，只把较短序列的 Query residual 移植到 L16，margin 从 -2.750
恢复到 +0.625；移植到 L20 后恢复到 +1.000。这说明中间 residual
漂移不仅与失败相关，而且足以因果地切换首 token 决策。

完整逐层表和最终读出结果位于：

- `outputs/onehop_layerwise_amplification_patch_20260728/exact_reconstruction/layerwise_exact_reconstruction.csv`
- `outputs/onehop_layerwise_amplification_patch_20260728/exact_reconstruction/final_readout_reconstruction.json`

---

## 3. 近期相关工作

### 3.1 主要路线

| 工作 | 核心方法 | 对本项目的启发与不足 |
|---|---|---|
| [Position Interpolation, 2023](https://arxiv.org/abs/2306.15595) | 将位置线性压回训练范围，并进行少量微调 | 减少 OOD 相位，但所有远程位置统一压缩，未直接分离顺序与语义 |
| [YaRN, ICLR 2024](https://openreview.net/pdf?id=wHBfxhZu1u) | 分频率插值并修正 attention temperature | Qwen3 长上下文的直接基础；仍然保留所有远程 token 的周期相位 |
| [LM-Infinite, NAACL 2024](https://aclanthology.org/2024.naacl-long.222/) | 限制远程有效距离并使用特殊稀疏 mask | 证明“不要让模型看到任意大的相对距离”有效，但远程 token 的区分能力有限 |
| [SelfExtend, ICML 2024](https://proceedings.mlr.press/v235/jin24b.html) | 近程 neighbor attention + 远程 grouped attention | 与“近程保序、远程降分辨率”高度一致；分组规则与语义相关性仍是分开的 |
| [Resonance RoPE, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.32/) | 调整波长，使训练范围和外推位置更一致 | 缓解位置 OOD，但没有直接消除远程相位对语义排序的翻转 |
| [Why Does Effective Context Length Fall Short?, 2024](https://arxiv.org/abs/2410.18745) | 指出训练中相对位置分布左偏，提出 STRING 平移有效位置 | 支持“标称上下文长度不等于真正训练充分的距离” |
| [LongRoPE2, ICML 2025](https://proceedings.mlr.press/v267/shang25a.html) | 以 needle PPL 搜索逐维缩放，并混合长短上下文训练 | 说明不同维度需要不同处理；需要搜索和继续训练 |
| [RNoPE, 2025](https://arxiv.org/abs/2501.18795) | 交替使用 RoPE 层和 NoPE 层，并结合局部/全局 attention | 强证据支持“全局语义不必每层携带 RoPE”；它是从训练开始设计的固定层间混合 |
| [HoPE, ACL 2025](https://aclanthology.org/2025.acl-long.1123/) | 将部分 RoPE 分量改成位置无关分量，只保留指定的高频旋转 | 与“位置子空间/内容子空间分离”直接相关；主要在从头训练的 3B 以内模型验证 |
| [HARPE, COLING 2025](https://aclanthology.org/2025.coling-main.326/) | 不同 head 使用不同 RoPE base | 支持 head-adaptive 设计；仍主要依赖固定的逐 head 频率与长上下文继续训练 |
| [RoPE Dimension Inefficiency, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.697/) | 发现大范围旋转使部分维度难以用于长程检索 | 支持按 head/维度选择位置强度，而不是所有维度共享一个策略 |
| [Understanding RoPE Extensions, COLING 2025](https://aclanthology.org/2025.coling-main.600/) | 从 attention 角度比较扩展方法 | 保持训练长度内的 attention pattern、降低 attention uncertainty 与检索正确率密切相关 |
| [LaMPE, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1608/) | 根据输入长度自适应地分配多粒度位置分辨率 | 说明固定映射不足，但其目标仍主要是位置重映射 |
| [Periodic RoPE, 2026](https://arxiv.org/abs/2605.27980) | 局部 SWA 层使用 RoPE，全局层使用 NoPE | 与“近程保序、远程去位置”最接近；采用固定的层间分工，没有在同一 head 内按检索置信度连续融合 |
| [TriAttention, 2026](https://arxiv.org/abs/2604.04921) | 利用跨位置稳定的 pre-RoPE Q/K 中心和三角级数估计 KV 重要性 | 证明 pre-RoPE 空间适合稳定选 Key；主要目标是推理期 KV 压缩，而非把语义分数直接注入 attention |
| [RoPE-ID / Frayed RoPE, ICLR 2026](https://research.ibm.com/publications/frayed-rope-and-long-inputs-a-geometric-perspective) | 只在部分通道施加高频 RoPE，以保持训练内 Q/K 簇分离 | 支持通道选择；其核心解释是 sink 功能和簇分离，而本文关注证据 QK、softmax 和 residual 因果链 |
| [ScoPE, ACL 2026](https://aclanthology.org/2026.acl-long.1650/) | 用不同 head 的指数级回看范围代替显式位置算术 | 表明顺序可由层级稀疏拓扑表达；但不是预训练 RoPE 模型的直接 retrofit |
| [RoPE Distinguishes Neither Positions Nor Tokens, 2026](https://arxiv.org/abs/2605.15514) | 理论证明长上下文下会出现 position/token inversion 与 aliasing | 说明仅调大 base 不能同时保住位置区分和 token 语义排序，需要结构性解耦 |

### 3.2 与我们的区别

已有方法大致分成三类：

1. 调整频率或位置映射；
2. 对远程位置截断、分组或降低分辨率；
3. 交替使用 RoPE 与 NoPE。

本项目更关注一个具体缺口：

> 对预训练好的模型，在每层每个 head 内，根据远程语义检索置信度，连续决定保留多少 post-RoPE 分数，并控制进入 softmax 的远程候选数量。

这个组合与 HoPE、RNoPE、Periodic RoPE、TriAttention 都有邻近关系，因此不能仅以“远程 NoPE”或“pre-RoPE 检索”作为创新点。可区分的部分必须是：

1. 同一 head 内的连续、置信度驱动门控，而非固定拆层；
2. pre-RoPE 负责远程候选，post-RoPE 保留训练内行为；
3. 逐 head 分布校准，使两个分数通道能进入同一个 softmax；
4. 候选预算与证据块位置修复联合设计；
5. 用 attention mass → residual → output margin 的因果指标训练或校准门控。

因此目标不是追求一个对所有 head、所有 token 都相同的新 base，而是把：

- 语义召回；
- 局部顺序；
- 分数尺度；
- softmax 候选预算；

放入同一个 attention 机制中。

---

## 4. Final：SAGE-RoPE 长程检索方案

SAGE-RoPE 的目标很简单：

$$
\boxed{
\text{近程保留 RoPE 顺序}
+
\text{远程增加语义召回}
+
\text{仅对约 2\% 候选做 softmax}
}
$$

它不删除预训练模型原有的 RoPE，而是为可能被相位压低的远程证据增加一条受控的 pre-RoPE 语义通道。当前可复现版本不需要重新训练模型。

### 4.1 第一步：计算两个分数

$$
s^R_{l,h}(t,j)
=
\frac{
\left(R_tq_{l,h}\right)^\top
\left(R_jk_{l,h,j}\right)
}{
\sqrt d
},
$$

$$
s^C_{l,h}(t,j)
=
\frac{
q_{l,h}^\top k_{l,h,j}
}{
\sqrt d
}.
$$

其中，$s^R$ 是标准 post-RoPE 分数，保留位置和顺序信息；$s^C$ 是 pre-RoPE 内容分数，不随证据—Query 距离发生额外旋转。

两个分数的尺度不同，因此在每层、每个 head 内，将语义分数校准到 post-RoPE 分布：

$$
\widehat{s}^{C}_{l,h}
=
\mu^{R}_{l,h}
+
\frac{
\sigma^{R}_{l,h}
}{
\sigma^{C}_{l,h}+\epsilon
}
\left(
s^{C}_{l,h}-\mu^{C}_{l,h}
\right).
$$

校准只使用当前 head 的均值和标准差，不需要证据标签。

### 4.2 第二步：选择约 2% 候选

$$
r_{l,h}(t,j)
=
\max\left(
s^R_{l,h}(t,j),
\widehat{s}^C_{l,h}(t,j)
\right).
$$

候选集合为：

$$
\mathcal C_{l,h}
=
\mathcal L_W
\cup
\mathcal S
\cup
\operatorname{TopK}
\left(
r_{l,h}(t,j),
\Delta>W
\right),
$$

其中：

- $\mathcal L_W$：最近 128 个 token，保留局部顺序；
- $\mathcal S$：前 16 个 sink/结构 token；
- 其余槽位：按双路分数 $r$ 从远程历史中选择；
- $|\mathcal C_{l,h}|\approx0.02N$。

逐 token 取两路最大值，是为了保留任一路认为重要的证据；只按 pre-RoPE 选择的版本作为候选消融。

### 4.3 第三步：保守融合并做稀疏 softmax

对入选的远程 token，当前验证版本使用：

$$
s^{\mathrm{SAGE}}_{l,h}(t,j)
=
0.75s^R_{l,h}(t,j)
+
0.25\widehat{s}^C_{l,h}(t,j).
$$

最近窗口仍使用原始 $s^R$；未入选 token 的分数设为 $-\infty$。最后只在候选集合内归一化：

$$
a_{l,h}(t,j)
=
\operatorname{softmax}_{j\in\mathcal C_{l,h}}
\left(
s^{\mathrm{SAGE}}_{l,h}(t,j)
\right).
$$

25% 是保守补偿：它足以救回一部分被 RoPE 相位压低的证据，同时保留 75% 原始 post-RoPE 几何，避免纯 pre-RoPE 改写预训练模型行为。

### 4.4 当前边界

已经端到端验证：

- pre-RoPE 远程候选召回；
- 逐层逐 head 分数校准；
- pre/post 双路候选；
- 75% post-RoPE + 25% pre-RoPE 融合；
- 约 2% 候选上的 sparse softmax。

尚未计入当前收益：

- 学习式逐 head 门控；
- 证据块位置修复；
- 近似向量索引带来的实际加速。

当前原型仍显式计算全历史 pre-RoPE QK，因此实验只证明方法质量，不代表已经获得端到端速度收益。

---

## 5. 扩展验证：24 个全新 seeds

### 5.1 资源约束

- Qwen3-8B，NF4 权重量化；
- seeds 8–31，共 24 条新样例，不再调整融合比例；
- 8K、16K、32K、64K，共 96 个“长度 × 样例”点；
- 每条样例包含两跳 clean 证据链和随机英文 filler；
- 所有方法复用同一条样例的公共 prefill；
- 每层每个 head 的候选预算约为上下文的 2%。

### 5.2 对照方法

实验实际运行了 8 个消融版本；正文只报告四个核心方法：

1. **Full RoPE：**标准 Full Attention；
2. **post-RoPE Top-2%：**按标准 attention logit 精确选择 2%；
3. **SAGE pre-only：**只用 pre-RoPE 分数选候选，再做 75%/25% 融合；
4. **SAGE dual-max：**用 pre/post 双路最大值选候选，再做 75%/25% 融合。

### 5.3 指标

- Gold evidence token recall；
- 两条证据链的 line hit / chain complete；
- 证据 attention mass；
- Gold PPL；
- 首答案 token 准确率；
- 相对 exact Top-2% 的逐样例 $\Delta$NLL。

PPL 是先平均 NLL 再取指数；置信区间采用 20,000 次 paired bootstrap。

### 5.4 核心结果

表中 PPL 后的括号是首答案 token 准确率；最后一列是 SAGE dual-max
相对 exact Top-2% 的逐样例平均 $\Delta$NLL：

| 长度 | Full RoPE | post-RoPE Top-2% | SAGE dual-max | $\Delta$NLL，95% CI |
|---:|---:|---:|---:|---:|
| 8K | 2.186（79.2%） | **1.929（75.0%）** | 2.119（75.0%） | +0.094 [-0.055, +0.271] |
| 16K | 5.974（41.7%） | **3.112（45.8%）** | 5.218（54.2%） | +0.517 [+0.285, +0.755] |
| 32K | 14.311（25.0%） | 8.056（20.8%） | **5.605（37.5%）** | **-0.363 [-0.593, -0.136]** |
| 64K | 5.868（37.5%） | 4.412（50.0%） | **3.914（45.8%）** | -0.120 [-0.282, +0.032] |

![24-seed SAGE-RoPE 扩展验证](../artifacts/20260730_local_global_rope_heldout24/analysis/heldout_comparison.png)

### 5.5 证据机制指标

| 长度 | 方法 | 证据 recall | 两链均命中 | 证据 mass |
|---:|---|---:|---:|---:|
| 8K | post-RoPE Top-2% | 32.5% | 64.9% | 5.2% |
| 8K | SAGE dual-max | 14.0% | 53.3% | 5.5% |
| 16K | post-RoPE Top-2% | 35.8% | 60.3% | 5.0% |
| 16K | SAGE dual-max | **40.0%** | **74.3%** | **5.3%** |
| 32K | post-RoPE Top-2% | 39.3% | 60.9% | 4.6% |
| 32K | SAGE dual-max | **46.8%** | **77.4%** | **5.0%** |
| 64K | post-RoPE Top-2% | 35.8% | 55.2% | **3.9%** |
| 64K | SAGE dual-max | **44.4%** | **76.2%** | 3.7% |

### 5.6 如何解释

- **32K 改善可靠。**PPL 下降 30.4%，准确率提高 16.7 个百分点，
  $\Delta$NLL 置信区间不跨 0。
- **64K 只有趋势。**PPL 和证据召回改善，但准确率没有提高，
  $\Delta$NLL 区间略跨 0，不能称为稳定胜出。
- **8K–16K 不应无条件启用。**短上下文中固定 128-token local window
  占用较多预算；16K 即使证据命中率上升，分数融合仍可能扰动不适合
  语义补偿的 head，使 PPL 变差。
- 因此，下一步最重要的不是继续提高统一融合比例，而是加入由距离、
  head 功能和检索置信度控制的回退门控。

---

## 6. 最终结论

RoPE 相位会改变远程证据的 QK 排名，而长上下文 softmax 又会稀释有限
的证据质量。SAGE-RoPE 同时增加内容召回并限制 softmax 候选，在 32K
获得了可靠改善；但它在短中程有副作用，在 64K 也尚未稳定提高准确率。

$$
\boxed{
\text{短程或低置信度：exact post-RoPE Top-2\%}
\quad
\text{长程且高置信度：SAGE dual-max}
}
$$

因此，当前结论不是“用 pre-RoPE 替代 RoPE”，而是“只在远程检索确有
收益时进行有限补偿”。学习式逐 head 门控和证据块位置修复仍是下一阶段，
不属于本轮已经验证的结果。

---

## 7. 复现实验与产物

主要脚本：

- 第一层精确相位分解：
  `scripts/run_first_layer_rope_phase_gpu7_20260730.sh`；
- 24-seed 扩展验证：
  `scripts/run_local_global_rope_heldout24_gpu7_20260730.sh`；
- 64K 并行分片：
  `scripts/run_local_global_rope_heldout64k_shard_20260730.sh`；
- 合并与统计：
  `src/merge_local_global_rope_heldout.py` 和
  `src/analyze_local_global_rope_heldout.py`。

本地完整产物：

- `artifacts/20260730_first_layer_rope_phase_gpu7/`；
- `artifacts/20260730_local_global_rope_heldout24/`。

扩展验证覆盖 24 seeds、4 个长度和 8 个方法，共 768 条记录；合并时发现
的 16 条并行重复记录内容一致并已按
`length × seed × variant` 去重。完整表、bootstrap 统计和图片均保存在
上述 artifact 中。
