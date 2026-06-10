# Round4 Linear Theory: Why Width Helps Low-Frequency Gradients

## 0. 要解释的问题

实验已经说明：

1. 高频/低频数据的效果差异由频率分布触发；
2. Zipf 训练中，高频数据长期主导真实梯度方向；
3. reweight / uniform fine-tune 可以恢复 tail，说明 tail 落后不是完全不可逆；
4. 宽模型对低频数据的改善显著大于高频数据。

还没有解释清楚的是：

> 为什么“宽度”这个结构量会特别有利于低频数据的梯度更新？

本文件用一个极简两层线性模型说明：即使函数类仍是线性的，**宽参数矩阵会改变梯度空间的几何结构**，让不同 feature 的梯度方向更接近正交，从而降低高频 feature 对低频 feature 的梯度干扰。

## 1. 一个最小模型

考虑输入只有若干离散 feature。为了简化，令 feature $i$ 的输入是 one-hot 向量：

$$
x_i = e_i,\qquad i = 1,\ldots,d.
$$

训练数据不是均匀分布，而是：

$$
P(i)=p_i.
$$

其中 head feature 的 $p_i$ 大，tail feature 的 $p_i$ 小。

模型是两层线性网络：

$$
f(x)=a^\top W x.
$$

其中：

$$
W\in\mathbb{R}^{m\times d},\qquad a\in\mathbb{R}^m.
$$

$m$ 是 hidden width。

对 feature $i$：

$$
f_i=f(e_i)=a^\top w_i.
$$

其中 $w_i$ 是 $W$ 的第 $i$ 列。

训练目标用平方损失：

$$
L=\frac{1}{2}\sum_i p_i(f_i-y_i)^2.
$$

记误差：

$$
\varepsilon_i=f_i-y_i.
$$

## 2. 宽度不改变函数类，但改变梯度几何

这个模型的函数本身是线性的：

$$
f(x)=\beta^\top x,\qquad \beta=W^\top a.
$$

所以如果只看可表达函数类，只要 $m\ge 1$，它已经能表达任意线性 $\beta$。从这个角度看，宽度似乎不重要。

但训练不是直接优化 $\beta$，而是优化参数 $(a,W)$。梯度下降看到的是参数空间中的梯度：

$$
\nabla_\theta f_i.
$$

宽度 $m$ 会改变不同 feature 的参数梯度是否互相重叠。

这就是关键：

> 宽度不是在这个 toy model 中增加线性函数表达能力，而是在增加参数空间中可供不同 feature 使用的近似独立梯度方向。

## 3. 每个 feature 的梯度方向

对 feature $i$：

$$
f_i=a^\top w_i.
$$

对参数求梯度：

$$
\frac{\partial f_i}{\partial a}=w_i,\qquad
\frac{\partial f_i}{\partial w_i}=a,\qquad
\frac{\partial f_i}{\partial w_j}=0\quad (j\ne i).
$$

所以 feature $i$ 的参数梯度可以理解成两部分：

$$
\nabla_\theta f_i=(w_i,\;0,\ldots,\underset{\text{column }i}{a},\ldots,0).
$$

不同 feature $i$ 和 $j$ 的梯度内积是：

$$
\langle \nabla_\theta f_i,\nabla_\theta f_j\rangle
=\langle w_i,w_j\rangle.
$$

因为 $w_i$ 和 $w_j$ 的 column-specific 部分不同，$a$ 那部分落在不同列上，彼此正交；它们唯一共享的交叉项来自 $a$ 参数上的梯度 $w_i$ 和 $w_j$。

而梯度范数是：

$$
\|\nabla_\theta f_i\|^2=\|w_i\|^2+\|a\|^2.
$$

因此 normalized gradient similarity 是：

$$
\cos_{ij}
=
\frac{\langle w_i,w_j\rangle}
{\sqrt{(\|w_i\|^2+\|a\|^2)(\|w_j\|^2+\|a\|^2)}}.
$$

这就是 feature-gradient interference 的最简单形式。

## 4. 宽度如何让 feature 梯度更正交

假设初始化为：

$$
a_r\sim\mathcal{N}\left(0,\frac{\sigma_a^2}{m}\right),\qquad
W_{r,i}\sim\mathcal{N}\left(0,\frac{\sigma_w^2}{m}\right).
$$

那么：

$$
\mathbb{E}\|a\|^2=\sigma_a^2,\qquad
\mathbb{E}\|w_i\|^2=\sigma_w^2,
$$

$$
\mathbb{E}\langle w_i,w_j\rangle=0\quad (i\ne j),\qquad
\operatorname{Var}(\langle w_i,w_j\rangle)=\frac{\sigma_w^4}{m}.
$$

因此对 $i\ne j$：

$$
\cos_{ij}=O_p\left(\frac{1}{\sqrt{m}}\right).
$$

也就是说：

> width 越大，不同 feature 的参数梯度越接近正交；交叉干扰项的随机幅度按 $1/\sqrt{m}$ 下降。

这是一个非常核心的性质。

窄模型中，不同 feature 的梯度方向会因为随机有限宽度而有较大 overlap。宽模型中，这些 overlap 会 concentration 到 0 附近。

## 5. 不均匀频率下的梯度更新

全量梯度为：

$$
g=\sum_i p_i\varepsilon_i\nabla_\theta f_i.
$$

对某个 tail feature $t$，一次梯度下降对它的预测值产生的函数空间变化为：

$$
\Delta f_t\approx -\eta\langle \nabla_\theta f_t,g\rangle.
$$

代入 $g$：

$$
\begin{aligned}
\Delta f_t
&\approx -\eta\sum_i p_i\varepsilon_i
\langle \nabla_\theta f_t,\nabla_\theta f_i\rangle \\
&= -\eta p_t\varepsilon_t\|\nabla_\theta f_t\|^2
-\eta\sum_{i\ne t}p_i\varepsilon_i
\langle \nabla_\theta f_t,\nabla_\theta f_i\rangle.
\end{aligned}
$$

第一项是 tail 自己的有效学习项：

$$
\text{self term}=p_t\varepsilon_t\|\nabla_\theta f_t\|^2.
$$

第二项是其他 feature 对 tail 的交叉影响：

$$
\text{cross term}
=\sum_{i\ne t}p_i\varepsilon_i
\langle \nabla_\theta f_t,\nabla_\theta f_i\rangle.
$$

对于 head feature $h$，$p_h$ 很大；对于 tail feature $t$，$p_t$ 很小。因此 tail 的 self term 天然弱，而 head 对 tail 的 cross term 可能很大。

这就是 Zipf 下 tail 难学的梯度动力学来源：

> tail 自己的更新权重小；head 的更新权重大；如果 head/tail 梯度不够正交，head 的更新就会在参数空间中不断干扰 tail。

## 6. 宽度如何改善 tail 的有效更新

根据上面的初始化结论：

$$
\frac{\langle \nabla_\theta f_t,\nabla_\theta f_h\rangle}
{\|\nabla_\theta f_t\|\,\|\nabla_\theta f_h\|}
=O_p\left(\frac{1}{\sqrt{m}}\right).
$$

所以 width 增大后，cross term 的相对幅度下降。tail 的更新更接近只由自己的 self term 决定：

$$
\Delta f_t\approx -\eta p_t\varepsilon_t\|\nabla_\theta f_t\|^2.
$$

虽然 $p_t$ 仍然小，但至少它不再被大量高频 feature 的随机交叉项淹没。

换句话说：

> 宽度并没有让 tail 样本变多；宽度让 tail 的梯度方向更干净，使得少量 tail 梯度能更稳定地积累。

这正对应实验中看到的现象：

- reweight 能提高 tail 的有效 $p_t$，所以 tail 恢复；
- uniform fine-tune 能提高 tail 的采样频率，所以 tail 恢复；
- 加宽能降低 head/tail 梯度交叉干扰，所以 tail 的边际改善更大；
- skew 越强，高频 cross term 越强，因此 width 的 tail-side value 越大。

## 7. 用 kernel 视角重写

定义神经切线核：

$$
K_{ij}=\langle \nabla_\theta f_i,\nabla_\theta f_j\rangle.
$$

在本模型中：

$$
K_{ii}=\|w_i\|^2+\|a\|^2,\qquad
K_{ij}=\langle w_i,w_j\rangle\quad (i\ne j).
$$

误差动力学近似为：

$$
\dot{\varepsilon}=-KP\varepsilon.
$$

其中：

$$
P=\operatorname{diag}(p_1,\ldots,p_d).
$$

如果 $K$ 是完全对角的：

$$
\dot{\varepsilon}_i=-K_{ii}p_i\varepsilon_i.
$$

每个 feature 独立学习。高频因为 $p_i$ 大而学得快，低频因为 $p_i$ 小而学得慢，但不会被其他 feature 干扰。

如果 $K$ 有较大 off-diagonal：

$$
\dot{\varepsilon}_i
=-K_{ii}p_i\varepsilon_i
-\sum_{j\ne i}K_{ij}p_j\varepsilon_j.
$$

tail feature $i=t$ 会受到 head feature 的强交叉项影响：

$$
\sum_{h\in\mathrm{head}}K_{th}p_h\varepsilon_h.
$$

因为 $p_h\gg p_t$，即使 $K_{th}$ 不大，也可能主导 tail 的早期变化。

宽度的作用是让：

$$
\frac{K_{ij}}{\sqrt{K_{ii}K_{jj}}}\rightarrow 0.
$$

因此训练动力学更接近 feature-wise decoupled learning。

## 8. 为什么低频改善大于高频

这个模型也解释了为什么加宽更改善 tail，而不是均匀改善所有 feature。

对 head feature $h$：

$$
\text{self term}=p_h\varepsilon_hK_{hh}.
$$

因为 $p_h$ 大，head 即使有一些 cross interference，自己的 self term 也足够强。它本来就能学得好。

对 tail feature $t$：

$$
\text{self term}=p_t\varepsilon_tK_{tt}.
$$

因为 $p_t$ 小，tail 的 self term 弱，所以它对 cross interference 更敏感。

因此当 width 增大、off-diagonal interference 下降时，最受益的是 tail：

- head: self term strong，interference 相对不致命；
- tail: self term weak，interference 一旦下降就明显改善。

这给出一个简洁解释：

> 宽度降低的是 feature 间的相对梯度混叠；低频 feature 的自更新项最弱，因此最受益。

## 9. 与实验结论的对应

### 9.1 高频/低频效果差异由频率导致

理论中学习速率含有 $p_i$：

$$
\dot{\varepsilon}_i\approx -K_{ii}p_i\varepsilon_i.
$$

所以 $p_i$ 越小，基础学习速度越慢。

### 9.2 高频主导梯度

全量梯度是：

$$
g=\sum_i p_i\varepsilon_i\nabla_\theta f_i.
$$

因此 head feature 由于 $p_i$ 大，在 $g$ 中权重更大。实验中的 positive alignment gap 正是这个式子的体现。

### 9.3 Reweight / uniform fine-tune 可恢复

Reweight 等价于把有效频率从 $p_i$ 改成更平衡的 $q_i$：

$$
g_{\mathrm{reweight}}
=\sum_i q_i\varepsilon_i\nabla_\theta f_i.
$$

Uniform fine-tune 等价于让：

$$
p_i=\frac{1}{d}.
$$

因此 tail self term 变大，tail 可以快速恢复。

### 9.4 宽度特别改善低频

宽度让：

$$
\frac{K_{ij}}{\sqrt{K_{ii}K_{jj}}}
=O_p\left(\frac{1}{\sqrt{m}}\right).
$$

tail 的 self term 小，所以最怕 off-diagonal 干扰；降低干扰后，tail 的有效更新改善最大。

## 10. 边界与注意事项

这个 toy theory 说明的是“宽度改善梯度几何”的一个最小机制，但它不是完整 LLM 理论。

需要保守的地方：

1. 两层线性模型没有 attention、非线性、token composition；
2. 这里主要分析 lazy / small-step regime 下的局部梯度几何；
3. 它解释的是 width 为什么能降低 feature-gradient interference，而不是直接证明所有真实大模型 scale law。

但它给 Round4 一个可检验的理论命题：

> 宽度提升会让 feature-gradient kernel 更接近对角化，降低高频 feature 对低频 feature 的交叉干扰；由于低频 feature 的自更新项本来最弱，它们从这种 decoupling 中获得最大边际收益。

## 11. 后续可验证预测

由这个理论推出三个直接预测：

1. 随 hidden width 增大，feature-gradient kernel 的 off-diagonal cosine 应下降，量级近似随 $1/\sqrt{m}$ 下降。
2. Zipf alpha 越大，tail 对 off-diagonal interference 越敏感，因此 width 的 tail-side improvement 越大。
3. Reweight 或 uniform fine-tune 会主要通过增大 tail self term 来恢复 tail，而不一定需要彻底改变 representation rank。

其中预测 2 已经被 Round4 alpha sweep 支持；预测 1 和 3 可以作为后续 Round5 的理论验证方向。
