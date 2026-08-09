# 问题：为什么上下文越长，大海捞针的正确答案置信度越低？

上下文变长时，通常同时发生两件事：

| 机制 | 发生了什么 | 直接结果 |
|---|---|---|
| **Softmax 竞争** | `N` 增大，更多 filler token 进入分母 | 真实证据的 attention mass 被稀释 |
| **RoPE 方向变化** | filler 插在证据与 Query 之间，使 `Δ` 增大 | 真实证据的 QK logit 在实验范围内总体下降 |

两项叠加，使证据写入 residual stream 的有效信号减弱，最终正确答案概率下降、PPL 上升。

```text
序列变长
├─ 候选 token 变多 ──> Softmax 分母变大 ──> 证据权重下降
└─ 证据离 Query 更远 ─> RoPE 相位改变 ────> 证据 QK 分数下降
```

## 1. Softmax：证据分数不变，也会被更多 token 稀释

真实证据的 attention 权重为：

$$
a_e=\frac{e^{s_e}}{e^{s_e}+\sum_{j\ne e}e^{s_j}}.
$$

若每个 filler token 的平均指数分数为 `C=E[e^{s_d}]`，则：

$$
a_e\approx\frac{e^{s_e}}{e^{s_e}+(N-1)C}.
$$

因此，即使 `s_e` 完全不变，`N` 增大也会让 `a_e` 下降。同时，候选越多，出现一个高分伪证据的概率也越高，真实证据的排名会进一步恶化。

这是一种**数量竞争**：不要求 Query 或真实证据 Key 的方向发生任何变化。

## 2. RoPE：距离变化会改变 Q 与 K 的相对方向

设 RoPE 前的内容向量为 `q、k`，Query 和证据位置分别为 `m、n`：

$$
q_m=R(m)q,\qquad k_n=R(n)k.
$$

经过 RoPE 后的点积为：

$$
s_e(\Delta)
=\frac{q_m^\top k_n}{\sqrt d}
=\frac{q^\top R(n-m)k}{\sqrt d},
\qquad \Delta=n-m.
$$

所以 RoPE 的位置影响只取决于**相对距离 `Δ`**。在第 `i` 个二维频率对中，其贡献可以写成：

$$
s_i(\Delta)=A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i).
$$

Head18 Layer11的q位置对于前面的k位置的Rope贡献波形图：

![image-20260721022045400](C:\Users\27814\AppData\Roaming\Typora\typora-user-images\image-20260721022045400.png)

单个频率对会周期性增强或削弱匹配，并不是距离越远就必然单调变差。但是一个 head 同时叠加多个不同频率，而且 Q/K 的权重由模型训练得到；当 `Δ` 离开模型熟悉的距离范围时，各频率更容易由相长叠加变为相消叠加。在我们的 Qwen3-8B 长距离实验范围内，最终表现为真实证据的 post-RoPE logit 和 cosine 总体下降。

## 3. Qwen3-8B 实验支撑

### 相对距离增长：RoPE 是 QK 方向损失的主要来源

在证据位于文本中部、Query 位于末尾的 8K→128K 反事实分解中：

| raw-logit 变化来源 | 贡献 |
|---|---:|
| Query 内容变化 | +0.349 |
| Key 内容变化 | +0.371 |
| **相对位置 / RoPE** | **−4.630** |

pre-RoPE 的内容匹配没有退化，真正的主要负项是相对距离带来的 RoPE 旋转。

### 相对距离固定：RoPE 退化消失，但 Softmax 稀释仍存在

把 evidence–Query 距离固定为 328，仅把前置 filler 从短上下文扩展到 128K：

| 指标 | 短上下文 | 长上下文 | 变化 |
|---|---:|---:|---:|
| evidence raw logit | 5.024 | 7.249 | +2.226 |
| head logsumexp | 14.486 | 17.048 | +2.562 |
| evidence attention mass | 0.723% | 0.437% | 降至 60.5% |
| Gold PPL 中位数 | 6.932 | 40.075 | 变坏 5.78× |

这证明：**固定距离可以保住 QK 方向，却不能阻止更多候选 token 扩大 Softmax 分母。**

作为补充，64K 的 16 样本实验中，post-RoPE Top-2% 将大部分竞争 token 删除后，保留的证据 attention mass 为 3.913%，接近 Full Attention 的 4.113%；正确答案 PPL 从 Full 的 1.317 改善到 1.202。这支持“无效竞争 token 会降低答案置信度”的解释。

## 结论

> 长上下文不是简单地让模型“忘记”证据，而是同时改变了证据的相对匹配分数和竞争环境。

1. **RoPE 决定能不能瞄准远处证据**：相对距离增长改变 QK 相位，使真实证据 logit 在当前任务和距离范围内总体下降。
2. **Softmax 决定证据能分到多少权重**：候选数量增长，即使证据 logit 不变，也会扩大分母并增加伪证据极值竞争。

**相对距离增长主要解释 QK 方向退化；候选数量增长独立解释 Softmax 稀释；二者叠加导致正确答案概率降低。**

---

## 附录：RoPE 旋转矩阵与二维频率贡献的推导

### A.1 二维旋转矩阵是什么

设 attention head 的维度为 `d`，RoPE 将这些维度组成 `d/2` 个二维频率对。第 `i` 个频率对在位置 `p` 的旋转矩阵为：

$$
R_i(p)=
\begin{bmatrix}
\cos(p\omega_i) & -\sin(p\omega_i)\\
\sin(p\omega_i) & \cos(p\omega_i)
\end{bmatrix}.
$$

其中 `ω_i` 是第 `i` 个二维对的角频率。标准 RoPE 通常使用：

$$
\omega_i=\theta_{base}^{-2i/d},
\qquad i=0,1,\ldots,d/2-1.
$$

Qwen3 在长上下文中可能通过 YaRN 等缩放策略调整实际使用的频率，但后面的推导不变：只需要把 `ω_i` 换成模型运行时的有效频率。

将所有二维旋转组合起来，可把完整旋转写成分块对角矩阵：

$$
R(p)=\operatorname{diag}\bigl(R_0(p),R_1(p),\ldots,R_{d/2-1}(p)\bigr).
$$

这里为了推导方便，把每个二维对写在相邻坐标中。Qwen3/Hugging Face 的实际张量布局采用 split-half 配对，即第 `i` 维与第 `i+d/2` 维组成一对；两种写法只差一次固定的坐标重排，数学含义完全相同。

旋转矩阵有三个重要性质：

$$
R(p)^\top R(p)=I,
\qquad R(p)^\top=R(-p),
\qquad R(a)R(b)=R(a+b).
$$

因此 RoPE 不改变 Q/K 的长度，只改变它们的方向。

### A.2 为什么 QK 点积只依赖相对位置

设 Query 位于 `m`，Key 位于 `n`，RoPE 前的内容向量为 `q、k`：

$$
q_m=R(m)q,
\qquad k_n=R(n)k.
$$

则旋转后的点积为：

$$
\begin{aligned}
q_m^\top k_n
&=(R(m)q)^\top R(n)k\\
&=q^\top R(m)^\top R(n)k\\
&=q^\top R(n-m)k.
\end{aligned}
$$

令相对位置 `Δ=n-m`，就得到：

$$
q_m^\top k_n=q^\top R(\Delta)k.
$$

这说明共同移动 Query 和 Key 不会改变 RoPE 位置项；真正起作用的是二者的相对距离。

### A.3 推导单个二维频率对的贡献

取第 `i` 个二维对，并记：

$$
q_i=
\begin{bmatrix}q_{i,x}\\q_{i,y}\end{bmatrix},
\qquad
k_i=
\begin{bmatrix}k_{i,x}\\k_{i,y}\end{bmatrix}.
$$

在相对位置 `Δ` 下，Key 被旋转为：

$$
R_i(\Delta)k_i=
\begin{bmatrix}
k_{i,x}\cos(\Delta\omega_i)-k_{i,y}\sin(\Delta\omega_i)\\
k_{i,x}\sin(\Delta\omega_i)+k_{i,y}\cos(\Delta\omega_i)
\end{bmatrix}.
$$

与 Query 做点积并展开：

$$
\begin{aligned}
q_i^\top R_i(\Delta)k_i
={}&(q_{i,x}k_{i,x}+q_{i,y}k_{i,y})\cos(\Delta\omega_i)\\
&+(q_{i,y}k_{i,x}-q_{i,x}k_{i,y})\sin(\Delta\omega_i).
\end{aligned}
$$

把 attention 的缩放因子也包含进系数，定义：

$$
A_i=\frac{q_{i,x}k_{i,x}+q_{i,y}k_{i,y}}{\sqrt d},
\qquad
B_i=\frac{q_{i,y}k_{i,x}-q_{i,x}k_{i,y}}{\sqrt d}.
$$

于是第 `i` 个二维频率对对 attention logit 的贡献就是：

$$
\boxed{s_i(\Delta)=A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i)}.
$$

整个 head 的 QK logit 是所有二维频率贡献之和：

$$
s(\Delta)=\sum_{i=0}^{d/2-1}s_i(\Delta).
$$

其中：

- `A_i、B_i` 由当前 token 的 Q/K 内容决定；
- `ω_i` 由 RoPE 频率配置决定；
- `Δ` 由证据与 Query 的相对位置决定。

还可以令：

$$
C_i=\sqrt{A_i^2+B_i^2},
\qquad
\phi_i=\operatorname{atan2}(B_i,A_i),
$$

从而得到等价形式：

$$
s_i(\Delta)=C_i\cos(\Delta\omega_i-\phi_i).
$$

这说明单个二维对是一条周期曲线，周期为 `2π/ω_i`；`C_i` 决定振幅，`φ_i` 是由 Q/K 内容决定的初始相位。一个 head 的整体曲线则是多个不同频率周期项的加权叠加，所以不会表现为一条简单、单调的距离衰减曲线。
