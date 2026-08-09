# 约 125M Qwen-style 模型的位置编码预训练实验

## 研究问题

在保持模型、数据、优化器和训练 token 数完全一致时，下面三种位置编码是否比原生 RoPE 更有利于长程检索，同时不破坏短程建模？

1. 浅层保留原生 RoPE，深层删除高频 RoPE。
2. 所有层保留 RoPE，但所有频率统一减速。
3. 在层和频率两个方向上连续减速，避免硬删除造成不连续变化。

本轮是机制筛选，不是最终论文结果。它只能回答这些操作在从头训练的约 128M Qwen-style 模型上是否值得继续；不能替代 4B/8B checkpoint 的继续训练验证。

## 可证伪猜想

- **H1：深层高频冗余。** 如果浅层已经写入局部顺序，深层的高频旋转会干扰远程内容匹配，那么只在深层删除 F0-F7 应提高长距离键值检索准确率，而且短距离损失不应明显变差。
- **H2：全局减速有利于外推。** 如果长程失败主要来自相位积累太快，那么令所有层的相位乘以 0.5，应把失败边界推向更长序列。
- **H3：连续层-频率缩放优于硬切换。** 如果局部顺序主要需要浅层高频，而远程内容匹配主要需要深层稳定方向，那么平滑缩放应同时优于原生 RoPE和硬删除。

## 模型

模型不是把 Qwen3-8B 的权重直接压缩，而是从头训练一个缩小的 Qwen-style decoder：

| 参数 | 数值 |
|---|---:|
| vocabulary size | 32,000 |
| hidden size | 768 |
| layers | 12 |
| query heads | 6 |
| KV heads | 2 |
| head dimension | 128 |
| SwiGLU intermediate size | 3,072 |
| tied input/output embedding | 是 |
| RoPE base | 1,000,000 |
| 参数量 | 约 128M |

128 维 head 保留了与 Qwen3-8B 相同的 64 个二维 RoPE 频率对，因此 F0-F7 等频带仍有相同的数学含义。

## 数学定义

第 \(l\) 层、第 \(i\) 个二维频率对的相位为

\[
\phi_{l,i}(p)=p\,\omega_i\,\alpha_{l,i},
\qquad
\omega_i=\theta^{-2i/d_h}.
\]

四个条件只改变 \(\alpha_{l,i}\)：

### 原生 RoPE

\[
\alpha_{l,i}=1.
\]

### 深层删除高频

\[
\alpha_{l,i}=
\begin{cases}
0,&l\ge 6\ \text{且}\ i<8,\\
1,&\text{其他情况}.
\end{cases}
\]

### 全层减速

\[
\alpha_{l,i}=0.5.
\]

### 平滑层-频率位置编码

\[
\alpha_{l,i}
=1-(1-\alpha_{\min})g_l(l)g_f(i),
\]

\[
g_l(l)=\sigma\!\left(\frac{l-l_c}{\tau_l}\right),
\qquad
g_f(i)=\sigma\!\left(\frac{i_c-i}{\tau_f}\right).
\]

本轮固定：

\[
\alpha_{\min}=0.25,\quad l_c=7.5,\quad \tau_l=1.5,
\quad i_c=7.5,\quad \tau_f=1.5.
\]

因此浅层和低频基本保持原生 RoPE；越靠后的层、越高的频率，旋转速度越平滑地接近 25%。该函数仍能写成每个 token 的独立旋转，所以不会增加 attention 的渐近复杂度。

## 实现合同

输入是长度固定的合成文档。每个文档包含 16 条随机 key-value 事实、可预测的局部文本结构和一个位于结尾的查询。模型必须在答案位置复制目标 value。

每个训练条件：

1. 使用相同随机种子初始化参数。
2. 每一步按相同步号和 rank 生成相同训练样本。
3. 只改变 `rope_variant`。
4. 保存训练 loss、短程检索、长度外推检索、Gold answer NLL、准确率、吞吐和 GPU 显存。
5. 若训练出现非有限 loss、不同条件的数据 hash 不一致或参数量偏离 115M-140M，则实验失败，不比较最终指标。

## 结论边界

合成检索成功只能说明该位置编码更适合受控的长距离关联复制。要形成论文主张，还需要自然语言继续预训练、WikiText/PG19 PPL、RULER、NIAH、LongBench，以及 4B/8B checkpoint 上的验证。

