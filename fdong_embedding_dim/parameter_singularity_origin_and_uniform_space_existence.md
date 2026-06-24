# 参数空间奇异性的来源、物理意义与更均匀学习空间的存在性问题

本文整理当前关于“大语言模型参数矩阵为什么出现大奇异值 / 大奇异方向，以及是否存在更高效、更均匀的参数/表征空间”的工作结论。

这里的逻辑按三个层次组织：

1. 事实：现在的大语言模型学出来的表征和参数空间是什么样。
2. 机制：这个事实如何由数据分布、activation 几何、梯度外积和模块功能选择共同产生。
3. 存在性：是否存在一种更均匀的参数/表征空间，使 long-tail 学习更高效。

最后列出 TODO：能否用 MoE 根据上述产生机制设计更高效的学习方法。

---

## 0. 三个层次的一句话核心回答

### 事实：现在的大语言模型学出来的表征和参数空间是什么样

现代 LLM 的 raw residual 表征空间会在早期形成少数稳定 dominant directions，后续层长期继承这些方向；参数空间中也会出现选择性的大奇异通道，但不是所有参数模块都对齐同一个方向，而是某些功能上有效的模块强烈吸收 input/output activation 的主方向。

### 机制：这个事实如何由数据分布、activation 几何、梯度外积和模块功能选择共同产生

Zipf 高频 pattern 首先在真实 token occurrence 的 activation matrix 中形成 common direction，Pre-Norm residual path 让该方向在 raw residual stream 中跨层继承，梯度外积把 activation 方向写入参数矩阵，而只有能利用该方向降低 loss 的模块会把它放大成最大奇异通道。

### 存在性：是否存在一种更均匀的参数/表征空间，使 long-tail 学习更高效

目前有初步存在性证据：reweighting 显示削弱 common direction 的早期垄断可以让 long-tail 学得更快、整体效率提高，说明高度奇异的 dense 学习路径不是唯一可行路径；但是否存在自然可训练的、更均匀参数/表征空间，仍需要 oracle split / MoE existence experiments 进一步证明。

---

## 1. 事实：现在的大语言模型学出了强各向异性的表征和参数空间

### 1.1 表征空间事实

在真实 Qwen3-0.6B 上，我们观察到 raw residual stream 的 layer input hidden states 具有非常强的主方向稳定性。

实验脚本：

- `fdong_embedding_dim/common_direction_experiments/run_qwen_layer_input_pc1_similarity.py`

结果：

- `fdong_embedding_dim/outputs/qwen_layer_input_pc1_similarity/qwen_layer_input_pc1_similarity.csv`

关键现象：

| transition | raw residual PC1 abs cosine |
|---|---:|
| layer 0 → 1 | 0.067 |
| layer 1 → 2 | 0.884 |
| layer 2 → 3 | 0.360 |
| layer 3 → 4 | 0.9999998 |
| layer 4 以后 | 基本接近 1.0 |

同时 raw residual top1 energy 从第 3 层开始非常高：

| layer | raw residual top1 energy |
|---:|---:|
| 0 | 0.080 |
| 1 | 0.052 |
| 2 | 0.049 |
| 3 | 0.9997 |
| 14 | 0.996 |
| 27 | 0.798 |

这说明：

> raw residual stream 在早期发生一次主方向转向后，后续层基本继承同一个 dominant direction。

需要注意：Q/K/V 实际吃到的是 `RMSNorm(h_l)`，不是 raw `h_l`。RMSNorm 后的 attention input PC1 仍会在中后层发生变化。因此“第 4 层后方向几乎不变”主要指 raw residual stream，而不是每层 Q/K/V 的实际输入。

### 1.2 参数空间事实

真实 Qwen3-0.6B 中，不同参数模块的最大奇异方向与对应 input/output activation 主方向存在选择性对齐。

实验脚本：

- `fdong_embedding_dim/common_direction_experiments/run_qwen_activation_parameter_alignment.py`

结果：

- `fdong_embedding_dim/outputs/qwen_activation_parameter_alignment/qwen_activation_parameter_alignment.csv`

对每个模块，我们抓取：

- module input activation \(X_{\text{in}}\)
- module output activation \(Y_{\text{out}}\)
- 参数矩阵 \(W\) 的输入侧最大奇异方向 \(v_1(W)\)
- 参数矩阵 \(W\) 的输出侧最大奇异方向 \(u_1(W)\)

比较：

\[
v_1(W) \leftrightarrow \operatorname{PC1}(X_{\text{in}})
\]

\[
u_1(W) \leftrightarrow \operatorname{PC1}(Y_{\text{out}})
\]

强对齐例子：

| layer | module | input align | output align |
|---:|---|---:|---:|
| 14 | `self_attn.k_proj` | 0.704 | 0.985 |
| 14 | `self_attn.v_proj` | 0.717 | 0.975 |
| 27 | `mlp.down_proj` | 0.095 | 0.438 |
| 27 | `mlp.gate_proj` | 0.011 | 0.790 |
| 27 | `mlp.up_proj` | 0.001 | 0.495 |

弱对齐例子：

| layer | module | input align | output align |
|---:|---|---:|---:|
| 21 | `self_attn.q_proj` | 0.00012 | 0.00015 |
| 21 | `self_attn.k_proj` | 0.00038 | 0.00006 |
| 21 | `self_attn.v_proj` | 0.00004 | 0.00006 |

这说明：

> 参数矩阵的大奇异方向不是所有模块都自动复制 activation common direction，而是选择性地出现在某些功能模块中。

---

## 2. 机制：参数空间的大奇异方向为什么出现

### 2.1 高频数据结构先进入 activation，而不是直接进入参数

静态 embedding table \(E\) 并不天然带 token occurrence frequency。每个 vocab row 只出现一次。

真正带频率的是 activation matrix：

\[
X_l =
\begin{bmatrix}
h_1 \\
h_2 \\
\cdots \\
h_N
\end{bmatrix}
\]

这里每一行是一个真实 token occurrence 的 hidden state。

如果数据中存在高频 token、短语、格式、句法 pattern，那么它们会在 \(X_l\) 中重复出现，形成 frequency-weighted activation geometry。

在合成 shared-K + Zipf 数据中，我们看到：

- `withK_zipf` 会显著提高 frequency-weighted attention input activation 的 top1 energy；
- 这比直接看静态 embedding table 更贴近“频率如何进入模型”。

因此更准确的机制起点是：

\[
\text{Zipf / high-frequency pattern}
\rightarrow
X_{\text{activation}} \text{ 出现 common direction}
\]

而不是：

\[
E \rightarrow W
\]

### 2.2 Pre-Norm residual path 让 raw common direction 跨层继承

现代 LLM 常用 Pre-Norm / RMSNorm：

\[
h_{l+1}=h_l+F_l(\operatorname{RMSNorm}(h_l))
\]

RMSNorm 作用在 \(F_l\) 的输入上，但 raw residual path：

\[
h_l \rightarrow h_{l+1}
\]

绕过了 RMSNorm。

所以如果某层 raw residual stream 已经有：

\[
h_l = A_l e_c + r_l
\]

且 \(A_l e_c\) 很大，那么后续层：

\[
h_{l+1}=h_l+\Delta h_l
\]

只要 \(\Delta h_l\) 不足以旋转主方向，raw PC1 就会继续保持 \(e_c\)。

这解释了 Qwen3-0.6B 中 raw residual PC1 从第 3/4 层后几乎不变的现象。

RMSNorm 不会删除 common direction，因为它不是 projection：

\[
h \mapsto h-\operatorname{Proj}_{e_c}(h)
\]

它只是 per-token scale normalization。

### 2.3 梯度外积把 activation 方向写入参数

对任意线性层：

\[
y = Wx
\]

单个样本的梯度是：

\[
\nabla W = g_y x^\top
\]

如果很多样本输入都有共同方向：

\[
x_s = a_s e_c + r_s
\]

那么 batch 梯度包含：

\[
\nabla W
=
\left(\sum_s a_s g_s\right)e_c^\top
+
\sum_s g_s r_s^\top
\]

第一项是 rank-1-ish 的，并且输入侧方向是 \(e_c\)。

因此 activation common direction 会通过梯度外积反复写入参数矩阵。

### 2.4 但不是所有模块都会吸收这个方向

数学只说明模块“看得到”这个方向，不说明模块一定会使用它。

模块是否形成大奇异方向，取决于：

\[
\text{该模块沿这个方向更新是否能降低 loss}
\]

因此真实条件是：

\[
X_{\text{input}} \text{ 有 common direction}
+
\text{模块能用该方向有效降 loss}
\Rightarrow
W \text{ 形成大奇异方向}
\]

Qwen3-0.6B 的 alignment 实验支持这个判断：某些模块强对齐，某些模块几乎不对齐。

这说明参数奇异性具有功能选择性。

---

## 3. 最大奇异方向和最大奇异值的物理意义

对参数矩阵：

\[
W=\sum_i \sigma_i u_i v_i^\top
\]

如果第一项占主导：

\[
W\approx \sigma_1 u_1 v_1^\top
\]

那么：

\[
Wx\approx \sigma_1 u_1(v_1^\top x)
\]

所以：

- \(v_1\)：该模块最强读取的 input hidden feature direction；
- \(u_1\)：该模块最强写出的 output feature direction；
- \(\sigma_1\)：这个 read/write 功能通道的增益强度。

因此，最大奇异方向的物理意义不是“某个抽象的坏方向”，而是：

> 该参数模块最高增益的输入-输出功能通道。

### 3.1 Attention 参数中的意义

对 \(W_q\)、\(W_k\)：

\[
q=W_qh_q,\quad k=W_kh_k
\]

attention score 是：

\[
q^\top k
=
h_q^\top W_q^\top W_k h_k
\]

定义：

\[
B_{QK}=W_q^\top W_k
\]

\(B_{QK}\) 表示 hidden space 中的有效 attention matching rule。

因此：

- \(W_q/W_k\) 的输入方向：attention routing 主要读取什么 residual feature；
- \(W_q/W_k\) 的输出方向：query/key space 中主要写出的匹配特征；
- \(B_{QK}\) 的大奇异方向：hidden space 中最强的 query-key matching channel。

对 \(W_v\)：

- 输入方向：value 模块主要读取什么 feature；
- 输出方向：被 attention 搬运的 dominant content feature。

对 \(W_o\)：

- 输入方向：attention head output 中主要被读取的方向；
- 输出方向：写回 residual stream 的主方向。

### 3.2 MLP 参数中的意义

对 MLP：

\[
z = W_{\text{up}}h,\quad
g = W_{\text{gate}}h,\quad
o = W_{\text{down}}(\phi(g)\odot z)
\]

最大奇异通道表示：

- up/gate 读取 residual stream 中什么 feature；
- up/gate 在 intermediate space 中制造什么 dominant feature；
- down 从 intermediate space 中读取什么 feature；
- down 写回 residual stream 的什么方向。

Qwen 实验中，很多 MLP 模块 output-side 对齐比 input-side 更强，说明它们不一定直接复制输入 common direction，而是在输出空间制造 dominant feature。

---

## 4. 机制总结：参数奇异性从哪里来

当前最准确的链条是：

\[
\text{高频数据结构}
\rightarrow
X_{\text{activation}} \text{ 出现 common direction}
\rightarrow
\text{Pre-Norm residual path 继承 raw common direction}
\rightarrow
\nabla W=\sum_s g_s x_s^\top
\rightarrow
\text{功能上有效的模块吸收该方向}
\rightarrow
\sigma_1(W) \text{ 变大，奇异向量稳定}
\]

所以参数空间奇异性的来源不是单一因素，而是：

1. 数据频率结构；
2. activation 各向异性；
3. residual stream 的跨层继承；
4. 梯度外积耦合；
5. 模块功能选择。

---

## 5. 反例与修正：只改 residual normalization 不够

我们测试过一个激进结构：

\[
h_{l+1}
=
\operatorname{RMSNorm}(h_l)
+
F_l(\operatorname{RMSNorm}(h_l))
\]

实验脚本：

- `fdong_embedding_dim/common_direction_experiments/run_stage8_normed_skip_synthetic.py`

结果：

- `fdong_embedding_dim/outputs/common_direction_causal/stage8_normed_skip_synthetic/summary.json`

这个结构确实降低了 raw 表征 top1 energy：

| metric | PreNorm | Normed-skip |
|---|---:|---:|
| final raw representation top1 energy | 0.510 | 0.486 |

但它没有降低参数空间奇异化，反而增强了：

| metric | PreNorm | Normed-skip |
|---|---:|---:|
| mean parameter top1 energy | 0.683 | 0.782 |
| max parameter top1 energy | 0.940 | 0.986 |
| mean parameter input alignment | 0.364 | 0.396 |
| mean parameter output alignment | 0.798 | 0.914 |

结论：

> common direction 不只是 Pre-Norm residual path 的副产物。即使切断 raw residual 的 scale 继承，只要任务仍然有 high-frequency/shared-K 结构，模型仍会在参数矩阵里重新制造高增益奇异通道。

这说明不能把目标简化成“压平 residual stream”。参数空间也需要被理解和干预。

---

## 6. 问题 2：是否存在更均匀且学习效率更高的参数/表征空间？

这里的问题是存在性，不是 solution。

我们要问：

> 是否存在一种训练路径或模型状态，使参数/表征空间的有效方向更均匀，同时 long-tail 学习更快、整体训练效率更高？

当前答案：

> 有初步存在性证据，但还没有完成真实 LLM 级别证明。

### 6.1 为什么理论上应当存在

如果表征空间被一个巨大 common direction 主导：

\[
h=Ae_c+r,\quad A\gg \|r\|
\]

那么优化器更容易利用 \(e_c\)，因为沿这个方向的小变化会带来较大的 logit、attention score 或 hidden state 改变。

结果是：

- common pattern 很快学会；
- residual / long-tail directions 更新收益低；
- long-tail 学习被延迟；
- 参数和表征的有效维度降低。

如果存在一种空间，使不同 feature directions 都有足够 scale 和可优化性，那么 long-tail feature 不必在 common direction 的阴影下学习。

因此更均匀空间的理论目标不是“消灭 common feature”，而是：

> 防止少数 high-frequency directions 垄断 scale、gradient 和参数功能通道。

### 6.2 reweighting 提供存在性线索

之前的 reweighting 实验显示：

- common direction 早期形成被削弱；
- embedding / residual update 的 effective rank 上升；
- gradient 对 common direction 的偏好下降；
- common 收敛变慢；
- long-tail 收敛变快；
- 整体训练效率提升。

这说明：

> 当前高度奇异的学习路径不是唯一可行路径。

存在一条更均匀、更有利于 long-tail 的学习路径。

注意：reweighting 不一定是最终方法。它只是一个存在性证据，说明模型可以不完全依赖单一 common channel 学会任务。

### 6.3 尚未证明的部分

我们还没有证明：

1. 真实 LLM pretraining 中一定存在自然可达的更均匀参数空间；
2. 更均匀的参数谱本身一定提升最终 validation loss；
3. 表征空间和参数空间可以同时更均匀且不损害 common feature；
4. MoE 一定能找到这样的空间。

normed-skip 的失败尤其说明：

> 压平表征不等于压平参数。模型可能把奇异性从 residual stream 转移到参数矩阵。

所以问题 2 目前只能说：

> 存在性有初步支持，但需要专门 oracle / controlled experiment 来证明。

---

## 7. TODO：能否用 MoE 根据上述机制提出更高效的学习方法

MoE 的合理动机不是“增加参数量”，而是：

> 改变 common feature 和 long-tail feature 是否必须共享同一个全局参数/表征空间。

基于上述机制，MoE 应当被设计成针对以下问题：

1. 高频 common direction 在 activation 中出现；
2. dense 参数模块把它吸收成高增益通道；
3. long-tail feature 被迫通过同一个通道学习；
4. residual / tail directions 的有效学习效率下降。

因此 MoE 的目标应当是：

\[
\text{common pressure 和 long-tail pressure 分配到不同专家/子空间}
\]

而不是只追求 routing entropy 或 load balance。

### 7.1 MoE 方法需要验证什么

如果 MoE 真按我们的机制工作，应观察到：

1. expert 内部的 activation top directions 不完全相同；
2. common-heavy examples 和 long-tail examples 不再强行共享同一参数 top singular channel；
3. expert 参数谱比 dense global 参数谱更局部、更均衡；
4. long-tail loss 下降更快；
5. common loss 不会严重退化；
6. 参数奇异性不会简单在每个 expert 内复现。

### 7.2 需要警惕的失败模式

MoE 可能失败于：

- routing collapse：所有 token 仍走同一专家；
- expert 内复现 dense 模型的 common direction；
- load balance 看似健康，但 feature specialization 不真实；
- 增加参数量带来收益，却不是因为 common/long-tail 解耦；
- long-tail expert 数据太少，反而学得更慢。

### 7.3 下一步存在性实验

为了回答“更均匀参数/表征空间是否存在”，MoE 之前可以先做 oracle 版本：

1. 按 known group / frequency band 做 oracle expert split；
2. 保持总 compute 或参数量可控；
3. 测 expert 内参数谱、activation 谱、long-tail learning speed；
4. 对比 dense baseline；
5. 判断更均匀空间是否真的提高学习效率。

如果 oracle split 都不能提升 long-tail 学习效率，那么 MoE 方向需要重新审视。

如果 oracle split 能提升，再研究可学习 routing 是否能逼近这个 oracle。

---

## 8. 当前结论

### 问题 1：参数大奇异方向如何产生，物理意义是什么？

基本可以回答：

> 参数矩阵的大奇异方向来自高频数据结构诱导的 activation common direction。该方向通过 Pre-Norm residual path 在 raw representation 中继承，并通过梯度外积写入参数矩阵。只有能利用该方向降低 loss 的模块会吸收并放大它。最大奇异方向的物理意义是某个参数模块最高增益的 feature read/write 功能通道。

### 问题 2：是否存在更均匀且学习效率更高的空间？

目前有初步存在性证据：

> reweighting 显示，削弱 common direction 的早期垄断可以让 long-tail 学得更快、整体效率提高。因此高度奇异路径不是唯一可行路径。

但尚未完成证明：

> 我们还需要 oracle split / MoE existence experiments 来证明，是否存在自然可训练的、更均匀的参数/表征空间，并且它在真实或更强合成任务上学习效率更高。
