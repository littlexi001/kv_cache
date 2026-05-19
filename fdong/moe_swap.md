# 推理友好的专家激活模式的 MoE 架构

## 第二阶段的背景与关键发现

混合专家（Mixture-of-Experts, MoE）是扩展模型容量的一类关键架构：通过条件计算，每个 token 只激活少量专家，从而在不按比例增加计算量的前提下获得更大的参数规模。然而，MoE 在推理时面临一个关键的显存瓶颈：尽管路由在 token 级别是稀疏的，但同一条序列中的不同 token 往往会共同激活大量专家。结果是，在推理过程中大多数专家仍需要常驻在 GPU 显存中，这限制了 MoE 在显存受限设备上的可部署性。

一个自然的解决思路是“专家换入换出”（expert swapping）：如果能够预测未来会被激活的专家，就可以提前预取并在 GPU 显存中换入/换出专家，从而减少需要常驻的专家集合。关键问题在于：路由是否足够可预测，以抵消权重传输带来的额外开销。

为回答这一问题，我们分析了 MoE gating 的决定因素。由于路由通常由线性投影 + softmax + top-$k$ 选择实现，它对 token 表示 $h_{l,t}$ 中的主导结构高度敏感。我们的经验与理论结果表明，隐藏激活具有强烈的谱各向异性：少数几个领先的奇异方向（“spike”子空间）解释了绝大部分激活能量，而 gate 的方向往往与该子空间对齐。因此，专家选择主要由低维 spike 特征决定，而不是由完整的高维表示决定。

在 Qwen1.5-MoE 上的实验进一步表明，这些 spike 成分主要反映了来自词嵌入并由残差连接在各层中保留的短程、低层次 token 与词法信息。这会使得相同 token 的路由模式相对稳定，从而在原理上支持“专家可预测”。

然而，从系统角度分析可知，专家 swapping 仍然强烈受带宽限制。即使预测准确，在现实设置下，由于 PCIe 带宽限制与系统开销，专家权重的传输时间仍可能比每层计算时间高出约 $2$-$3\times$。因此，第二阶段得到的是一个“混合结论”：路由可预测，但朴素的细粒度 swapping 仍然昂贵。这促使我们同时考虑系统感知的 swapping 策略，以及从架构层面设计更“天生显存友好”的专家激活模式。

## 语言模型表征空间中的特征结构

我们研究大语言模型（LLM）表征空间的内在结构，试图识别具有物理意义的性质，以指导模型优化。

我们的出发点是假设：表征空间中的特征分布满足三条理想性质。第一，每个单独的特征本身应是高维的，即它仍在与模型内部表示相同的高维空间中表达。第二，每个数据点应呈现低秩的组合结构，即它可由少量特征的组合表示。第三，跨数据点的特征组合应是稀疏的，即不同样本由不同的特征子集刻画，而不是共享一个稠密叠加的特征集合。

这些假设的动机有两点：其一，若数据由高维空间中的任意高秩组合生成，则模型要高效学习并组织这些知识将缺乏可解释性与可行性；其二，若数据由大量特征的稠密组合表示，不同数据点之间的差异会被显著削弱，表征空间将难以保留具有判别性的结构。

### 特征的定义

#### 基于 SVD 的定义

一种自然的特征定义方式是：对 token 表示矩阵进行奇异值分解（SVD），将奇异向量视为特征方向；当 token 在该方向上投影非零（或显著）时，认为其激活了该特征。

然而，经验观察挑战了这一定义。在一个 $1024$ 维的表征空间中（即有 $1024$ 个奇异向量），我们发现每个 token 平均会激活约 $400$ 个特征。如此密集的激活并不符合“低维组合”的直觉，从而与我们的先验假设相矛盾。

#### 基于稀疏字典的定义

为进一步检验是否存在稀疏、低维的特征结构，我们采用稀疏字典学习框架。具体地，我们联合学习：

- 一组基向量（字典原子）。
- 用于重构 token 表示的稀疏系数。

并施加约束，使每个 token 最多激活 $k$ 个基向量。

在字典训练中，学习到的字典 $D$ 与稀疏编码矩阵 $A$ 通过近似最小化如下目标进行优化：

$$\min_{D,A} \frac{1}{2}\|X - AD\|_F^2 + \alpha \|A\|_1,$$

其中 $X$ 是训练 token 表示矩阵，$D$ 由字典原子组成，$A$ 为对应的稀疏系数矩阵。第一项鼓励对训练表示的精确重构，$\ell_1$ 惩罚项鼓励系数稀疏。

作为对照实验，我们也将表示约束为最多使用从 SVD 得到的 $k$ 个奇异向量。结果显示，在相同的 $k$ 下，学习到的稀疏字典相比 SVD 基线能实现显著更低的重构误差。这为“数据确实存在低维组合结构，但 SVD 不能捕获合适的特征基”提供了有力证据。

### 现有模型中特征表征的性质

我们分析了 SVD 特征与稀疏字典特征在几何与统计性质上的差异：

1. 激活概率分布：在稀疏字典定义下，各特征被激活的概率大致相近；而 SVD 特征呈现偏斜分布，领先奇异向量被激活得更频繁。
2. 角度分布：稀疏字典特征近似正交（并非严格正交），平均两两夹角约为 $84^\circ$，说明特征在高维空间中分布较为均匀。

![按句子统计的基向量使用率。](figures/layer_1_sorted_sentence_mean_usage.png)

图 1：按句子统计的基向量使用率。


## 模型架构

### 概述

为了支持面向推理的显存优化，我们提出层级化的 Common-Unique MoE（H-MoE），通过显式的多级专家层次结构来产生稳定的常驻专家集合，并最小化 swapping 的数据负载。

![H-MoE 模型架构](figures/model_arch.png)

图 2：层级化 Common-Unique MoE（H-MoE）模型架构。

考虑一个带有 MoE 前馈模块的 Transformer 层。与其将每个 token 路由到一个扁平的专家池，H-MoE 将专家组织为多个层级。在每个层级 $i$，我们引入一个对每个 token 都“必经”的公共专家 $f^{(i)}_{\mathrm{com}}(\cdot)$（始终激活），同时引入 $n_i$ 个唯一专家 $\{f^{(i)}_{j}(\cdot)\}_{j=1}^{n_i}$。对 token 表示 $h$，层级 $i$ 的 router 会选择 $k_i$ 个唯一专家（top-$k_i$ 路由），但公共专家始终包含在计算图中。因此，层级 $i$ 的计算可以写为：

$$y^{(i)}(h) = f^{(i)}_{\mathrm{com}}(h) + \sum_{j\in \mathcal{S}_i(h)} f^{(i)}_{j}(h), \qquad |\mathcal{S}_i(h)| = k_i,$$

其中 $\mathcal{S}_i(h)\subseteq \{1,\dots,n_i\}$ 表示被选中的唯一专家索引集合。

该设计确保每个 token 始终经过一条稳定的“主干路径”（所有层级的公共专家），而条件计算则由轻量的唯一专家承担。其关键性质在于：所有层级的公共专家都是强制激活的，路由仅决定额外激活哪些唯一专家。


### Synthetic 实验：层级稀疏特征下，Hierarchical MoE 是否更能捕捉到真实特征分布？

#### 1. 数据构造

我们对真实的数据做了建模，我们认为真实数据中的信息/ feature 如下假设：

- 有些 feature 几乎所有数据都会用到，例如语法、句法、常见结构；
- 有些 feature 只在某一大类数据中出现，例如“数学类”数据会激活数学相关 feature；
- 更细的 feature 只在最具体的数据类中出现，例如“抽象代数”和“数学分析”虽然都属于数学，但它们激活的细粒度 feature 不同。

<!-- > 具体我们建模这种性质的数据集（h,y）为： -->
具体我们建模这种性质的数据集（h,y）为：

每条数据的输入向量由三部分相加得到：
$$
h = h_{\mathrm{global}} + h_{\mathrm{middle}} + h_{\mathrm{leaf}} + \epsilon .
$$
三部分分别表示：

| 部分                    | 激活规则               | 含义                             |
| ----------------------- | ---------------------- | -------------------------------- |
| $h_{\mathrm{global}}$ | 所有数据都激活         | 所有数据共享的 common feature    |
| $h_{\mathrm{middle}}$ | 同一个中层类的数据共享 | 大类 feature，例如“数学”         |
| $h_{\mathrm{leaf}}$   | 每个最细类不同         | 最细粒度 feature，例如“抽象代数” |

其中激活概率满足：

$$
q_G > q_M > q_L
$$

每一个feature的激活幅度符合正态分布，也就是：越 common 的 feature，被越多数据看到。
因此，每条数据只激活一小部分真实 feature，但不同数据类激活的 feature 组合不同。这也自然的得到：公共 feature 的激活频率更高，公共部分贡献的二阶能量更大因此数据矩阵普空间出现各向异性的原因。

输出也按同样的层级生成：
$$
y = A_{\mathrm{path}}h + \eta,
$$
其中
$$
A_{\mathrm{path}}
= A_{\mathrm{global}} + A_{\mathrm{middle}} + A_{\mathrm{leaf}}.
$$


---

#### 2. 为什么 Hierarchical MoE 应该更适合这个 setting？

普通 MoE 直接的gating部分接收到的是完整的表征：
$$
h = h_{\mathrm{global}} + h_{\mathrm{middle}} + h_{\mathrm{leaf}}.
$$
但 $h_{\mathrm{global}}$ 和 $h_{\mathrm{middle}}$ 方差更大、出现更频繁。它们会主导输入的整体变化。

如果我们真正想区分的是最细类，那么关键信号其实在 $h_{\mathrm{leaf}}$。普通 MoE 的 gate 要在完整输入里找这个较弱的 leaf 信号，因此更容易被 common 部分干扰。

可以把线性区分的信噪比粗略写成：
$$
\text{SNR}_{\mathrm{flat}}
\approx
\frac{\|\Delta_{\mathrm{leaf}}\|}
{\sqrt{\operatorname{Var}(h_{\mathrm{global}})+\operatorname{Var}(h_{\mathrm{middle}})+\operatorname{Var}(h_{\mathrm{leaf}})}} .
$$
分层 MoE 先把 common 部分单独处理掉，再让细粒度 expert 主要面对剩余的局部信号。此时区分 leaf 的信噪比更接近：
$$
\text{SNR}_{\mathrm{hier}}
\approx
\frac{\|\Delta_{\mathrm{leaf}}\|}
{\sqrt{\operatorname{Var}(h_{\mathrm{leaf}})}} .
$$
因为分母中少了大量共享但不区分最细类的方差，所以：
$$
\text{SNR}_{\mathrm{hier}} > \text{SNR}_{\mathrm{flat}}.
$$
从函数学习角度也是一样的。真实机制是：
$$
A_{\mathrm{path}}
= A_{\mathrm{global}} + A_{\mathrm{middle}} + A_{\mathrm{leaf}}.
$$
普通 MoE 往往需要每个 expert 重复学习 common 部分；分层 MoE 可以把 common 部分共享掉，只让最细 expert 学自己的 leaf 部分。

所以当数据真的具有这种层级 feature 结构时，分层 MoE 更容易实现：

1. 不同 feature 的数据类被分到不同 expert；
2. 不同 expert 的参数学习不同真实 feature。

而我们在小实验的验证下也发现了：Hierarchical MoE 出现明显对角结构：

![普通 MoE 学习不同 feature 的情况。](figures/normal_moe_feature_expert.png)

![Hierarchical MoE 学习不同 feature 的情况。](figures/hier_moe_feature_expert.png)


- 横轴：真实 feature group；
- 纵轴：expert / path；
- 颜色：该 expert 参数对该真实 feature group 的使用强度。


$$
\text{one expert} \leftrightarrow \text{one leaf feature group}.
$$

普通 MoE 则是混合的：

$$
\text{one expert} \leftrightarrow \text{many feature groups}.
$$

这说明：在这种层级稀疏 feature setting 下，Hierarchical MoE 不只是 loss 更低，也确实更接近我们想要的“不同 expert 学不同 feature”。



| model | test MSE ↓ | leaf routing NMI ↑ | leaf purity ↑ | parameter feature diversity ↑ |
|---|---:|---:|---:|---:|
| Flat MoE | 0.00430 | 0.335 | 0.383 | 0.059 |
| Hierarchical MoE | **0.00125** | **0.997** | **0.999** | **0.796** |
| Ground-truth Tree MoE | 0.00078 | 1.000 | 1.000 | 0.773 |
| Random routing | 0.00451 | 0.027 | 0.410 | 0.000 |

---

### 真实数据实验结果

我们的实验结果表明，Hierarchical MoE 在优化效率上仍具有竞争力。具体而言，在总参数预算匹配且小于 $0.5$B 的设置下，层级 MoE 的训练损失仅略高于标准扁平 MoE 基线，如下图所示。这意味着在低参数规模下，该层级结构引入的约束并未显著降低模型容量，同时为推理时的显存管理提供了潜在优势。后续工作将扩展更大规模的实验，并进行层级结构选择的消融研究，以及训练质量与推理效率之间权衡的系统评估。

![在总参数预算低于 $0.5$B 且匹配时的训练损失曲线示意图。层级 MoE 与扁平 MoE 基线非常接近，仅存在较小的损失差距。](figures/hier_moe_loss.png)

图 2：在总参数预算低于 $0.5$B 且匹配时的训练损失曲线示意图。层级 MoE 与扁平 MoE 基线非常接近，仅存在较小的损失差距。

---

### 显存节省与 swapping 负载的潜在降低

二阶段工作识别里带宽瓶颈：专家 swapping 之所以昂贵，是因为按 token 传输整套专家权重通常比在设备端进行计算要慢得多。而 Hierarchical-MoE 层级化设计通过结构性拆分专家，提供了降低 swapping 负载的机制：

1. 始终开启的公共专家可长期常驻 HBM。
2. 轻量的唯一专家按需 swap。

为说明潜在收益，考虑一个两级层次结构示例。设第 1 级 common 专家是较大的 MLP（例如 $d\!\rightarrow\!2d\!\rightarrow\!d$），而第 2 级 unique MoE 被组织为 $G=4$ 个组。每个组包含一个 common 专家（例如 $d\!\rightarrow\!d\!\rightarrow\!d$）以及多个本地 unique 专家（例如 $d\!\rightarrow\!d/2\!\rightarrow\!d$），其中每个 token 只激活一个（top-1）。在该设定下，每个 token 的激活参数规模近似为：

$$4d^2 \; + \; 2d^2 \; + \; d^2 \; = \; 7d^2,$$

这与计算量可比的总参数量 $28d^2$ 扁平 MoE 基线的“激活参数预算”一致。

关键优势来自“常驻集合”。在 H-MoE 中，第 1 级公共专家（$4d^2$）以及所有组级公共专家（$4\times 2d^2 = 8d^2$）可以永久常驻，从而常驻参数规模仅为：

$$12d^2.$$

与此同时，只有最小的本地唯一专家需要 swap，每个 token 的 swapping 负载约为：

$$d^2.$$

相比之下，若扁平 MoE 基线在专家无法完全常驻的情况下，需要 swap 的通常是整套被激活专家权重（约 $7d^2$）。因此，在该示例下，H-MoE 将每 token 的 swapping 负载降低了：

$$\frac{7d^2}{d^2} = 7\times,$$

同时仍保持相对较小的常驻占用。总体而言，该层级化结构在 (i) HBM 常驻占用 与 (ii) PCIe/HBM 传输成本 之间提供了更有利的平衡，使得在带宽受限的推理场景中，专家 swapping 更可行，也更利于在显存受限设备上部署 MoE。
