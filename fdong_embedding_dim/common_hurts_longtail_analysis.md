# Common 数据如何影响 Longtail 学习：从 Toy Bigram 到真实 LLM 的梯度动力学分析

## 1. 背景：数据的长尾分布是天然先验

在自然语言中，数据呈现显著的长尾分布，表现在多个层面：

- **Token 层面**：高频 token（如 "the", "and", "a"）占据了训练数据的绝大部分，而大量低频 token（如 "quantum", "photosynthesis"）出现次数极少。
- **Domain 层面**：某些领域（如通用对话、新闻）数据量远大于长尾领域（如专业法律、医学文本）。
- **抽象 Feature 层面**：尽管我们尚不能精确刻画每个 feature，但可以确信概念/语法模式/"风格"等抽象特征同样服从长尾分布。

**本文的核心前提**：数据的长尾分布不是可调的超参数，而是人类语言世界的固有属性。我们不去质疑它——我们去理解它如何塑造模型的内部表征。

## 2. 核心问题：Common 是否影响 Longtail 的学习？

我们关注三个递进的问题：

1. **是否存在影响？** 高频（common）数据的梯度更新是否会改变低频（longtail）数据的学习动态？
2. **影响的方向是什么？** Common 加速还是拖慢了 longtail 的收敛？
3. **为什么会有这样的影响？** 梯度动力学层面的机制是什么？

我们在一个极度简化的 toy bigram 模型上系统回答了这三个问题，然后讨论其向真实 Transformer 的推广。

## 3. Toy 实验设置

### 3.1 原始实验：4 Group 循环 Bigram

**模型**：12 个 token，分为 4 个 group（A=common, B/C/D=tail）。每个 group 的 3 个 token 形成循环 bigram：

$$A_0 \to A_1,\; A_1 \to A_2,\; A_2 \to A_0$$

Bigram 预测模型：$E, W \in \mathbb{R}^{12 \times 2}$（独立训练的两张参数表），$\text{logit}_j = E_i \cdot W_j$。

**数据分布**：
- Uniform：A=25%, B=25%, C=25%, D=25%
- Zipf：A=70%, B=10%, C=10%, D=10%

**初始化**：spread —— 四个 group 分别放在 2D 平面的 0°, 90°, 180°, 270° 方向上。

### 3.2 新增实验：K-token 连接器模型

在真实语言中，"and"等高频 token 并非自身构成一个独立的语义 group——它们是连接各个概念的**桥接 token**。为建模这一现象，我们设计了 K-token 实验：

- 13 个 token：1 个 K（连接器）+ 4 个 group × 3 个 token
- Bigrams：$G_0 \to G_1,\; G_1 \to K,\; K \to G_2,\; G_2 \to G_0$（每个 group 有 2 条内部转移 + 2 条经过 K 的转移）
- K 出现在 50% 的 bigram 中（作为 target 或 input）
- Uniform 分布：A=B=C=D=25%

**核心假设**：K 会扮演类似 "and" 的角色——成为 embedding 空间的引力中心，所有其他 token 的表征被拉向 K。

## 4. 实验结果

### 4.1 现象：Common 确实拖慢了 Longtail 的收敛

| 指标 | Uniform (1:1:1:1) | Zipf (7:1:1:1) |
|------|-------------------|-----------------|
| 全体 acc=100% 步数 | **30** | **50** |
| Common 收敛步数 | 30 | 33 |
| Tail 收敛步数 | 30 | 44-50 |
| Final $\sigma_1/\sigma_2$ | 1.000 | 1.084 |

在 Zipf 分布下，尽管 common 和 tail 的初始化方向正交，tail 的收敛仍然被显著拖慢（50 vs 30 步）。

### 4.2 核心发现：Tail 在训练中被 Common 推离了初始正交方向

训练结束时各 group 质心偏离初始方向的角度：

| Group | Uniform (step 200) | Zipf (step 200) |
|-------|-------------------|-----------------|
| Common | **1.82°** | **2.08°** |
| Tail1 | **1.82°** | **40.85°** |
| Tail2 | **1.82°** | **41.10°** |
| Tail3 | **1.82°** | **1.96°** |

在 Uniform 下，所有 group 对称偏离（所有 1.82°），正交性保持。在 Zipf 下，**Tail1 和 Tail2 被推飞了 41°**，而 Common 几乎没动（2°）——Tail 的正交性被动态打破。

跨 group 余弦也证实了这一点：

| 余弦对 | Uniform | Zipf |
|--------|---------|------|
| Common·Tail1 | 0.000（正交）| **-0.681**（不再正交）|
| Tail1·Tail2 | -1.000（完全相反）| **-0.140**（几乎不再相反）|

### 4.3 K-token 实验：K 成为 embedding 空间的 mean bias

在 Uniform K-token 实验中：

```
K initial direction    = (+0.7070, +0.7070)   [45°]
Mean of non-K centroid = (+0.7076, +0.7066)   [误差 0.05°]
```

**所有非 K token 的 embedding 质心精确指向 K 的初始方向。** K 扮演了 embedding 空间的"引力中心"——这正是真实 LLM 中高频 token（如 "and"）作为 embedding mean bias 现象的 toy 级复现。

## 5. 机制分析：Common 如何影响 Longtail —— 一条完整的因果链

### 5.1 梯度做了什么？

Cross-entropy loss 对 bigram $i \to j$ 的梯度具有"相互吸引力"结构：

$$\frac{\partial L}{\partial W_j} = (\text{softmax}_j - 1) \cdot E_i$$

因为 $\text{softmax}_j < 1$（模型还不完美），系数为负 → **梯度下降把 $W_j$ 拉向 $E_i$**。对称地，$\frac{\partial L}{\partial E_i}$ 把 $E_i$ 拉向 $W_j$。

对于循环 bigram（如 $A_0 \to A_1 \to A_2 \to A_0$），这形成一个闭合吸引力环，使同 group 的所有 E 和 W 互相聚团，norm 持续增长。

### 5.2 跨组拉力从何而来？

关键：$W$ 矩阵（输出嵌入）出现在**所有 12 个 logit 中**。当处理 common 数据（如 $A_0 \to A_1$）时：

$$\frac{\partial L}{\partial W_{B0}} = \text{softmax}_{B0} \cdot E_{A0}$$

**Common 数据把 Tail 的 $W_{B0}$ 拉向 common 的 $E_{A0}$ 方向！**

### 5.3 两股力在拔河

每个 tail token 的 $W$ 向量同时承受两股力：

| | 组内力（来自本组 bigram） | 跨组力（来自 common 数据） |
|---|---|---|
| 方向 | Tail 自身的方向（y 轴） | Common 的方向（x 轴） |
| 频率权重 | 10% | **70%** |
| 净效果 | 想保持正交 | **把 Tail 拽向 x 轴** |

**7 倍的拉力差 → Tail 的 $W$ 向量被整体拽向 common 方向 → Tail 的循环链被污染 → 整个 Tail group 失去正交性 → 梯度重叠 → SIR 暴跌 → 有效学习率归零。**

### 5.4 反馈闭环

$$\text{频率不对称} \to \text{更新速度不对称 (7:1)} \to \sigma_1 \neq \sigma_2 \to \text{参数空间变形} \to \text{梯度不再正交} \to \text{SIR 暴跌 (157→0.6)} \to \text{Tail 学得更慢} \to \text{谱形变更严重}$$

Uniform 下此闭环从未启动（所有 group 等速推进），Zipf 下从第 1 步就开始运转。

### 5.5 K-token 实验揭示了"mean bias"的形成机制

在 K-token 设置中：

- **K 作为 target**（$G_1 \to K$）：$W_K$ 被拉向所有 $E_{G1}$ 的方向 → 合力指向全体 token 的质心
- **K 作为 input**（$K \to G_2$）：$E_K$ 被拉向所有 $W_{G2}$ 的方向 → 合力同样指向质心
- **对称拉力的平衡** → K 的方向 ≈ 全体 token 的均值方向

所有非 K token（尤其是低频 token）都通过 $G_1 \to K$ 的梯度被持续拉向 K 的方向。久而久之，K 的初始位置成为了整个 embedding 空间的引力中心。

## 6. 从 Toy 到真实 LLM 的推广

### 6.1 Toy 与 Transformer 的映射

| Toy 概念 | 真实 LLM 对应 |
|----------|--------------|
| Common group A（70% 频率） | "and", "the", "a" 等高频 function word |
| Tail groups B/C/D（10%） | 低频 content word（"quantum" 等）|
| Group 内部循环 bigram（$A_0 \to A_1$） | Token pair 之间的预测关系 |
| Common 的跨组拉力 → Tail 被推歪 41° | 低频 token embedding 被拉向 "and" 方向 |
| K 成为全体 token 的"引力中心" | "and" embedding 是 token embedding 的第一主成分方向 |

### 6.2 核心论断

> 在基于 cross-entropy 的 next-token prediction 训练范式下，高频 token 作为更多训练样本的 target，其输出嵌入被拉向全体前文 token 的质心；对称地，其输入嵌入也被拉向全体后文 token 的质心。当这两个质心接近时（"and" 的前后分布都非常多样化），高频 token 的 embedding 方向就稳定在全体 token embedding 的均值方向上。低频 token 由于出现次数少、前文固定，被这个均值方向持续吸引——这就是 embedding mean bias 的动力学起源。

### 6.3 "and" 为什么比 "the" 更典型？

| Token | 前文多样性 | 后文多样性 | 作为"均值"的资格 |
|-------|-----------|-----------|-----------------|
| **and** | 极高（几乎任何词后都可以接） | 极高 | ✅ 最佳代表 |
| **the** | 高 | 中等（后面通常是名词） | ✅ 较好，但有偏向 |
| **a** | 高 | 固定（后面必须是名词） | △ 有偏向 |

"and" 的前文和后文分布都极其均匀，使得拉力的所有方向分量几乎完全抵消 → "and" 最接近纯粹的均值。

## 7. 结论

1. **Common 确实影响 Longtail 学习**。这种影响不是统计巧合，而是 cross-entropy + softmax 梯度动力学的必然结果。

2. **影响的方向是负面的**。Common 的高频梯度更新通过跨组拉力（$W$ 矩阵作为共享输出空间的机制）将 tail 的 embedding 拉离其最优方向，导致 tail 的 SIR 暴跌、有效学习率归零。

3. **影响机制的核心**是 next-token prediction 训练范式本身：高频 token 是更多训练样本的 target → 它们的输出嵌入被拉向全体 token 的质心 → 低频 token 被推向这个质心 → embedding mean bias 形成。这不是初始化或超参数问题，是**训练目标的结构性后果**。

4. **K-token 实验**为上述论断提供了直接的 toy 级证据：在引入一个连接所有 group 的桥接 token K 后，K 的位置精确地成为了全体 token embedding 的质心，完美复现了真实 LLM 中 "and" 作为 embedding mean bias 的现象。

5. **向 Transformer 的推广**需要注意 LayerNorm 对 norm 增长的限制以及多层非线性变换提供的"逃生通道"，但底层机制（梯度把上下文表示拉向预测目标，频率高的拉得更狠）是相通的。
