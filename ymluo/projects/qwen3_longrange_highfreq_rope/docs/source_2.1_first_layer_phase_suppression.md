# 2.1 RoPE 的高频分量使长程语义匹配随距离失稳

## 1. 核心观点

> **高频 RoPE 会让长程语义匹配随距离失稳。**

标准 RoPE 为每个 attention head 配置同一组从高频到低频的位置旋转。

当一个 head 承载长程、粗粒度的语义关系时，这些旋转会把原本应当随距离保持稳定的语义匹配转化为振荡的 QK 匹配，并在长距离处形成深低谷。

这里的“失稳”不是指 RoPE 直接删除 residual stream 中储存的语义信息，而是指语义关系本身保持不变时，模型通过 QK 内积识别并读取该关系的能力仍会随相对距离变化。

## 2. 核心假设

上述观点依赖两个需要明确检验的假设：
- 长程语义应当对距离变化保持稳定
- 长程语义应当分布在其所在 head 的各维度上，而非仅分布在个别维度上。

### 2.1 长程语义的位置稳定性

自然语言中的关系具有不同的位置尺度。局部、细粒度关系依赖精确的位置变化，而长程关系通常对应实体、事实、主题或证据之间的高层语义联系。

本文认为长程语义应当具有“在一定 context 范围内，具有同长程语义的 token (head) 可被 q-k 匹配”的性质。

### 2.2 长程匹配没有被最低频方向主导

在所研究的模型中，长程匹配能量没有以决定性比例集中在目标窗口内近似不旋转的少数最低频 2d pair 上。

这一假设不要求匹配能量集中在高频方向，只排除模型已经把几乎全部长程匹配隔离到最低频方向的情形。从机制直觉上看，只要匹配能量没有被近似静止的最低频方向主导，其余多个旋转频率就可能通过相位叠加改变整体 QK 匹配。

## 3. Transformer 与 RoPE 的结构基础

本节使用标准 Transformer attention 与 RoPE 的两个结构性质。

### 3.1 Attention 通过 QK 匹配识别相关信息

Transformer 通过 attention 中的 QK 内积衡量 Query 与候选 Key 的匹配程度，并据此决定读取哪些 Value。

因此，对于一个原本能够被正确识别的长程 Query–evidence 关系，相关 attention head 在应用 RoPE 之前应当具有较高的 QK 内容匹配。

后文将归一化 pre-RoPE Q/K alignment 接近1作为“该关系原本能够被当前 head 识别”的数学表达，而不是关于自然语言或模型能力的额外假设。

### 3.2 所有 head 共享同一组 RoPE 频率

标准 RoPE 不根据 layer 或 head 承担的信息尺度调整频率配置，而是在每个 head 中使用同一组由高到低的旋转频率。

模型可以通过学习 Q/K 投影来调整语义信息在各个 2d pair 上的分布，但 RoPE 本身不提供与不同 layer、head 或语义尺度相适应的频率选择机制。

QK 匹配机制决定了长程关系如何被读取，而统一的 RoPE 频率配置决定了这种匹配如何受到相对位置旋转。

## 4. 主要理论结论

我们考虑一个负责识别 Query–evidence 长程关系的 attention head。

该 head 对相关 Query 和 evidence Key 的内容匹配表现为较高的 pre-RoPE QK alignment，而其匹配能量分散在多个 2d pair 上，没有被少数近似不旋转的最低频方向主导。

在这些条件下，post-RoPE QK alignment 一致接近一个由 RoPE 频率配置决定的多频 cosine trace。

对于足够大的 RoPE base $\theta$，在 $[0,c\theta]$ 的有限相对位置范围内（$c>0$ 为任意固定比例），该 trace 必然存在一个近零位置。

> **当一对 Q/K 原本高度对齐，且其 2d-pair 能量接近均匀分布、因而不由少数近似不旋转的最低频方向主导时，RoPE 的多频叠加会在有限相对距离内形成深低谷，使原本很高的 QK 匹配下降到接近零；在特定配置下，该匹配还会变成负数。**

对 Qwen3-8B 的 standard RoPE 配置 $(\theta=10^6,m=64)$，等能量理想 trace 的直接计算给出：

- 在 $0.1\theta=100{,}000$ 内，最低值为 $-0.12727$，位置为 $n=99{,}018$；
- 在 $0.14\theta=140{,}000$ 内，最低值为 $-0.17890$，位置为 $n=128{,}312$；
- 若用 $\varepsilon=0.005$ 表示 Q/K 的不完全对齐，用 $\eta=0.05$ 表示能量对等权模板的偏离，则在 140K 范围内可以保证 normalized QK alignment 从至少 $+0.995$ 降到至多 $-0.0289$。

这一结果解释了为什么短窗口中的正常检索不能保证长窗口中的稳定检索：短距离可能尚未覆盖多频叠加形成的深低谷，而更长的 context window 会暴露此前未出现的破坏性相位组合。

由于 140K 主实验使用 YaRN-$4\times$，standard RoPE 的上述数值不能直接对应实验低谷；该对应关系需要基于运行时 effective frequencies 重新计算。

本节聚焦以下机制链路：

$$
\text{固定的 pre-RoPE 内容对齐}
\quad\longrightarrow\quad
\text{随位置变化的 post-RoPE QK-score 抑制}.
$$

从 QK-score 抑制到 evidence attention mass 下降，以及该变化在后续层中的传播，将在 2.2 及后续分析中讨论。

## 5. 研究范围与术语

我们只分析一个 Query head 内的一对 evidence Query--Key。

本文把由同一 RoPE 频率共同旋转的二维坐标子空间称为
**rotary pair**，正文中简称 **2d pair**。当 rotary dimension 为
$d=2m$ 时，一个 head 中共有 $m$ 个 2d pair。Qwen3-8B 的 head
dimension 为 $d=128$，因此 $m=64$。

对于 base 为 $\theta>1$ 的 standard RoPE，第
$i\in\{0,\ldots,m-1\}$ 个 2d pair 使用频率

$$
\omega_i=\theta^{-2i/d}=\theta^{-i/m}.
$$

RoPE 后的 QK 内积只依赖相对位置。不失一般性，把 evidence Key
的位置看作零，把 Query 相对于它的位置记为整数 $n$。相对旋转算子为

$$
R_n=\operatorname{diag}
\left(R_{n\omega_0},\ldots,R_{n\omega_{m-1}}\right),
$$

其中 $R_\phi$ 表示旋转角为 $\phi$ 的二维旋转矩阵。

定义归一化 Query 和 Key：

$$
\widehat q=\frac{q}{\lVert q\rVert},
\qquad
\widehat k=\frac{k}{\lVert k\rVert}.
$$

它们在相对位置 $n$ 上的 post-RoPE alignment 为

$$
\rho(n)=\widehat q^\top R_n\widehat k.
$$

对应的 pre-softmax QK logit 为

$$
s(n)=\alpha\rho(n),
\qquad
\alpha=\frac{\kappa^2\lVert q\rVert\lVert k\rVert}{\sqrt d},
$$

其中 $\kappa$ 是具体实现使用的 rotary scale。

其他上下文 token 不会影响这一对 Q/K 自身的 logit；它们只会在 attention softmax 归一化时成为竞争项。

## 6. 核心条件的数学形式化

### 条件 A1：pre-RoPE Q/K 高度对齐

考虑在应用 RoPE 之前高度相似的 Query 和 evidence Key：

$$
\widehat q^\top\widehat k\geq1-\varepsilon,
\qquad
0\leq\varepsilon<1.
$$

对于一个可由 attention 识别的 Query–evidence 关系，相关 head 应当具有较高的 QK 内容匹配。

A1 将这一检索关系写成可用于误差分析的数值条件。

由单位向量的性质可得

$$
\begin{aligned}
\lVert\widehat q-\widehat k\rVert^2
&=
\lVert\widehat q\rVert^2+\lVert\widehat k\rVert^2
-2\widehat q^\top\widehat k\\
&\leq2\varepsilon,
\end{aligned}
$$

因此

$$
\boxed{
\lVert\widehat q-\widehat k\rVert
\leq\sqrt{2\varepsilon}.
}
$$

### 条件 A2：2d-pair 能量接近均匀模板

把归一化 Query 写成 $m$ 个 2d pair：

$$
\widehat q=(\widehat q_0,\ldots,\widehat q_{m-1}).
$$

定义第 $i$ 个 2d pair 上的能量

$$
w_i=\lVert\widehat q_i\rVert^2.
$$

因为 $\widehat q$ 已归一化，

$$
w_i\geq0,
\qquad
\sum_{i=0}^{m-1}w_i=1.
$$

在所研究的模型中，长程匹配能量没有以决定性比例集中在目标窗口内近似不旋转的少数最低频 2d pair 上。

为了获得显式的确定性误差界，我们使用“真实能量分布接近均匀模板”作为这一现象的充分条件。

定义均匀模板：

$$
u_i=\frac1m.
$$

我们用 $L_1$ 距离刻画真实能量分布与均匀模板的偏离：

$$
\boxed{
\lVert w-u\rVert_1\leq\eta.
}
$$

这一条件蕴含“匹配能量没有决定性地集中在最低频 2d pair”。

它不要求匹配能量集中在高频 2d pair，而允许能量自然地分散在从高频到低频的大量方向上。

因此，A2不要求高频方向支配匹配，只排除少数近似静止的最低频方向支配全部匹配。

这一条件不要求每一个 2d pair 内的 $q_i,k_i$ 都单独高度相似。

低能量 pair 可以存在较大偏差。

A1控制的是所有 pair 的总失配；A2控制的是理想 phase trace 的能量不能被少数最低频方向主导。

“接近均匀”是用于建立当前误差界的充分条件，而不是该机制成立所必需的最弱条件。

## 7. 化约为正权重 cosine mixture

令

$$
e=\widehat k-\widehat q.
$$

根据 A1，

$$
\lVert e\rVert\leq\sqrt{2\varepsilon}.
$$

于是

$$
\begin{aligned}
\rho(n)
&=\widehat q^\top R_n\widehat k\\
&=\widehat q^\top R_n(\widehat q+e)\\
&=\widehat q^\top R_n\widehat q+\widehat q^\top R_ne\\
&=\sum_{i=0}^{m-1}w_i\cos(n\omega_i)+r(n),
\end{aligned}
$$

其中

$$
r(n)=\widehat q^\top R_ne.
$$

由于 $R_n$ 是正交矩阵，

$$
\begin{aligned}
|r(n)|
&\leq
\lVert\widehat q\rVert\lVert R_ne\rVert\\
&=
\lVert e\rVert\\
&\leq\sqrt{2\varepsilon}.
\end{aligned}
$$

因此

$$
\boxed{
\rho(n)
=
\sum_{i=0}^{m-1}w_i\cos(n\omega_i)+r(n),
\qquad
|r(n)|\leq\sqrt{2\varepsilon}.
}
$$

这一步是 A1 带来的核心简化：一般 Q/K pair 的 RoPE score 是带有任意
content phase 的 cosine--sine mixture；高 Q/K alignment 使它统一接近
一个系数非负的 cosine mixture。

定义由 RoPE 配置完全决定的均匀模板 trace：

$$
F_{\theta,m}(n)
=
\frac1m\sum_{i=0}^{m-1}
\cos\left(n\theta^{-i/m}\right).
$$

真实权重与均匀模板之间的差异满足

$$
\begin{aligned}
\left|
\sum_i(w_i-u_i)\cos(n\omega_i)
\right|
&\leq
\sum_i|w_i-u_i|\,|\cos(n\omega_i)|\\
&\leq
\lVert w-u\rVert_1\\
&\leq\eta.
\end{aligned}
$$

结合 A1 的 content mismatch 误差，对任意整数位置 $n$ 都有

$$
\boxed{
|\rho(n)-F_{\theta,m}(n)|
\leq
\eta+\sqrt{2\varepsilon}.
}
$$

这就是后续定理的关键桥梁：

> 高 Q/K 内容对齐把真实 score 化为正权 cosine mixture；分散的
> 2d-pair 能量进一步使该 mixture 在所有位置上一致接近一个只由
> $(\theta,m)$ 决定的确定性 trace。

## 8. 给定配置下的有限窗口相位抑制定理

给定整数搜索窗口 $0\leq n\leq N$，定义

$$
\mu(\theta,m,N)
=
\min_{0\leq n\leq N}F_{\theta,m}(n).
$$

当 $\mu<0$ 时，定义正的 phase-suppression margin

$$
\gamma(\theta,m,N)=-\mu(\theta,m,N)>0.
$$

### 定理 1：有限窗口内的 phase suppression

在 A1--A2 成立时，存在一个整数位置
$n^\star\in\{0,\ldots,N\}$，使

$$
\boxed{
\rho(n^\star)
\leq
\mu(\theta,m,N)+\eta+\sqrt{2\varepsilon}.
}
$$

由于 $\rho(0)\geq1-\varepsilon$，alignment 的下降幅度至少为

$$
\boxed{
\rho(0)-\rho(n^\star)
\geq
1-\varepsilon-\mu(\theta,m,N)
-\eta-\sqrt{2\varepsilon}.
}
$$

如果 $\mu=-\gamma<0$，则

$$
\boxed{
\rho(0)-\rho(n^\star)
\geq
1+\gamma-\varepsilon-\eta-\sqrt{2\varepsilon}.
}
$$

对应的 QK logit 下降至少为

$$
\boxed{
s(0)-s(n^\star)
\geq
\alpha
\left[
1+\gamma-\varepsilon-\eta-\sqrt{2\varepsilon}
\right].
}
$$

此外，只要总近似误差小于理想 trace 的负 margin：

$$
\boxed{
\eta+\sqrt{2\varepsilon}<\gamma,
}
$$

就能保证真实 post-RoPE alignment 为负：

$$
\rho(n^\star)<0.
$$

#### 证明

取 $n^\star$ 为有限集合上 $F_{\theta,m}$ 的最小值点。由上一节的
一致逼近界，

$$
\begin{aligned}
\rho(n^\star)
&\leq
F_{\theta,m}(n^\star)
+\eta+\sqrt{2\varepsilon}\\
&=
\mu(\theta,m,N)+\eta+\sqrt{2\varepsilon}.
\end{aligned}
$$

再从 $\rho(0)\geq1-\varepsilon$ 中减去上述上界，得到 alignment
下降界。乘以与位置无关的正数 $\alpha$，得到 QK logit 下降界。

当 $\mu=-\gamma$ 且
$\eta+\sqrt{2\varepsilon}<\gamma$ 时，

$$
\rho(n^\star)
\leq
-\gamma+\eta+\sqrt{2\varepsilon}
<0.
$$

证毕。

## 9. Base 为 $10^6$ 时的确定性计算

本节数值使用：

- standard RoPE；
- $\theta=10^6$；
- $m=64$；
- 2d-pair energy 均匀；
- 整数相对位置；
- 不使用 YaRN 或其他 frequency scaling。

### 9.1 在 $N=100{,}000=0.1\theta$ 的窗口内

对所有整数位置进行穷举，得到

$$
\mu(10^6,64,100000)
=
-0.12727017393640289,
$$

最小值位置为

$$
n^\star=99018.
$$

因此理想 normalized alignment 从 $F(0)=1$ 下降到
$F(99018)=-0.12727$，理想下降幅度为 $1.12727$。

考虑一组示例误差参数：

$$
\varepsilon=0.005,
\qquad
\eta=0.05.
$$

总扰动上界为

$$
\eta+\sqrt{2\varepsilon}
=0.05+0.1
=0.15.
$$

因此

$$
\rho(99018)
\leq
-0.12727+0.15
=0.02273.
$$

初始 alignment 至少为 $\rho(0)\geq0.995$，所以可保证下降至少

$$
\rho(0)-\rho(99018)
\geq
0.995-0.02273
=0.97227.
$$

这个误差预算能保证在 $0.1\theta$ 内几乎抹除原始 alignment，但不能
保证最终 alignment 为负。要保证负值，需要

$$
\eta+\sqrt{2\varepsilon}<0.12727.
$$

### 9.2 在 $N=140{,}000=0.14\theta$ 的窗口内

穷举得到

$$
\mu(10^6,64,140000)
=
-0.1789006636660568,
$$

最小值位置为

$$
n^\star=128312.
$$

在该位置：

- 64 个等权 2d pair 中有 39 个产生负贡献；
- 负向贡献总和为 $-0.4546669$；
- 其余 pair 的正向贡献总和为 $+0.2757662$；
- 合计为 $-0.1789007$。

仍取 $\varepsilon=0.005$、$\eta=0.05$，则

$$
\rho(128312)
\leq
-0.1789007+0.15
=-0.0289007.
$$

下降幅度至少为

$$
\begin{aligned}
\rho(0)-\rho(128312)
&\geq
0.995-(-0.0289007)\\
&=
1.0239007.
\end{aligned}
$$

因此，在这组示例条件下，normalized QK alignment 可以被严格保证从至少 $+0.995$ 降到至多 $-0.0289$。

### 9.3 复现代码

~~~python
import numpy as np

theta = 1e6
m = 64
N = 140_000
omega = theta ** (-np.arange(m) / m)
w = np.ones(m) / m

best_value, best_position = 2.0, None
for start in range(0, N + 1, 20_000):
    positions = np.arange(start, min(start + 20_000, N + 1))[:, None]
    values = np.cos(positions * omega) @ w
    index = int(values.argmin())
    if values[index] < best_value:
        best_value = float(values[index])
        best_position = start + index

print(best_value, best_position)
~~~

这些数字是固定有限和的确定性浮点计算，而非统计估计。如果论文最终把
这个负 margin 当作形式化证书，应再使用高精度或区间算术验证浮点舍入误差。

## 10. $\theta$ 足够大时的一般结论

上一节的定理把给定模型配置的问题化约成了可计算量
$\mu(\theta,m,N)$。下面证明另一个更一般的结果：

> 对任意固定比例 $c>0$，当 base $\theta$ 足够大时，在
> $[0,c\theta]$ 内必然存在一个位置，使均匀模板 alignment
> 任意接近零。

固定 $c>0$，令

$$
N=\lfloor c\theta\rfloor.
$$

对一个频率 $\omega$，定义它在离散窗口内的均值

$$
D_N(\omega)
=
\frac1{N+1}\sum_{n=0}^{N}\cos(n\omega).
$$

由有限几何级数

$$
\sum_{n=0}^{N}e^{in\omega}
=
\frac{1-e^{i(N+1)\omega}}{1-e^{i\omega}},
$$

可得

$$
\boxed{
|D_N(\omega)|
\leq
\min\left\{
1,
\frac{1}{(N+1)|\sin(\omega/2)|}
\right\}.
}
$$

有限序列的最小值不大于其平均值，因此

$$
\begin{aligned}
\min_{0\leq n\leq N}F_{\theta,m}(n)
&\leq
\frac1{N+1}\sum_{n=0}^{N}F_{\theta,m}(n)\\
&=
\frac1m\sum_{i=0}^{m-1}D_N(\omega_i).
\end{aligned}
$$

进一步定义

$$
B_m(\theta,c)
=
\frac1m\sum_{i=0}^{m-1}
\min\left\{
1,
\frac{1}{
(\lfloor c\theta\rfloor+1)
|\sin(\theta^{-i/m}/2)|
}
\right\}.
$$

于是

$$
\boxed{
\min_{0\leq n\leq\lfloor c\theta\rfloor}
F_{\theta,m}(n)
\leq
B_m(\theta,c).
}
$$

当 $m,c$ 固定且 $\theta\to\infty$ 时，$B_m(\theta,c)$ 中每一项都
趋于零。即使对最慢频率 $i=m-1$，也有

$$
\begin{aligned}
&(\lfloor c\theta\rfloor+1)
\sin\left(\frac{\theta^{-(m-1)/m}}2\right)\\
&\qquad\asymp
\frac c2\theta^{1/m}
\longrightarrow\infty.
\end{aligned}
$$

因此

$$
\boxed{
B_m(\theta,c)\longrightarrow0
\qquad
(\theta\to\infty).
}
$$

### 定理 2：base 足够大时，在固定比例窗口内存在近零位置

对任意固定的 $m\in\mathbb N$、$c>0$ 和 $\zeta>0$，都存在

$$
\theta_0=\theta_0(m,c,\zeta),
$$

使得对所有 $\theta\geq\theta_0$，都存在一个整数
$n^\star\leq c\theta$ 满足

$$
\boxed{
F_{\theta,m}(n^\star)\leq\zeta.
}
$$

在 A1--A2 下，真实 Q/K alignment 满足

$$
\boxed{
\rho(n^\star)
\leq
\zeta+\eta+\sqrt{2\varepsilon}.
}
$$

因此 alignment 下降至少为

$$
\boxed{
\rho(0)-\rho(n^\star)
\geq
1-\varepsilon-\zeta-\eta-\sqrt{2\varepsilon}.
}
$$

#### 证明

由 $B_m(\theta,c)\to0$，可以选择足够大的 $\theta_0$，使得所有
$\theta\geq\theta_0$ 都满足

$$
B_m(\theta,c)\leq\zeta.
$$

有限序列的最小值不大于其均值，而均值不大于
$B_m(\theta,c)$，所以必然存在整数 $n^\star\leq c\theta$，使

$$
F_{\theta,m}(n^\star)\leq\zeta.
$$

最后应用第 7 节的一致逼近界，并从
$\rho(0)\geq1-\varepsilon$ 中相减，即得结论。证毕。

### 一般定理的结论边界

这个均值论证保证：当 $\theta$ 足够大时，可以在 $c\theta$ 内把理想 alignment 压到任意接近零。

该结论不包含统一的负常数

$$
F_{\theta,m}(n^\star)\leq-\gamma.
$$

要得到严格负 margin，有两种路线：

1. 对给定配置直接计算 $\mu(\theta,m,N)$，如第 9 节；
2. 在指定的 base 区间上建立更强的 phase-coverage 定理。

因此，一般结论与给定配置下的结论具有不同强度：

- **一般大-base 定理**：保证 near-erasure；
- **固定 Qwen 配置的确定性计算**：在 100K--140K 内实际得到负值。

## 12. 适用范围

本节刻画的是 RoPE 对单个 Query–evidence QK 匹配的相位抑制，其结论适用于满足 A1 和 A2 定量条件的 Q/K pair。

以下问题需要由后续理论或实验进一步连接：

- 不同 layer 和 head 中满足 A1、A2 的长程语义关系占比；
- retrieval-relevant heads 的真实频率能量模板，以及它们在不同任务和距离下的稳定性；
- 单个 evidence logit 的下降如何在 softmax 竞争中转化为 attention probability 的下降；
- attention mass、Value 写入和后续 Query state 如何将局部匹配低谷传播为最终答案失败；
- standard RoPE 的理论低谷与 YaRN-$4\times$ 运行时有效频率下的实验低谷之间的对应关系。

## 13. 经验验证

1. 对预先确定的 evidence–Query Q/K pair 测量 $\varepsilon=1-\cos(q,k)$，避免仅分析事后筛选的样本。
2. 测量真实 rotary-pair energy vector $w$，以及它到均匀模板和 held-out 经验模板 $\bar w$ 的距离。
3. 比较 retrieval-relevant heads 与 random/distractor pairs 满足能量分散条件的程度。
4. 使用论文 YaRN-$4\times$ 140K protocol 的运行时有效频率重新计算 $\mu$。
5. 固定 pre-RoPE Q/K 并显式改变相对旋转，检验理论预测的低-score位置。
6. 控制 support、竞争 logits、Values 和后续 Query states，区分 phase-only 最低点与自然 answer-failure 位置。

## 14. 理论扩展

- 用 learned retrieval-head frequency template 替换均匀模板，并在 held-out Q/K pairs 上证明扰动界；
- 给出有限且有实际意义的 $\theta_0(m,c,\zeta)$，而不只证明渐近存在；
- 在明确的 base 区间内，使用解析方法或带认证的区间计算建立统一负 phase margin；
- 把A2从“接近模板”弱化为“slow-band mass上界 + spectral anti-concentration”；
- 研究更强的 per-2d-pair alignment 条件能否把 $\sqrt{2\varepsilon}$ 误差改进为更紧的界。
