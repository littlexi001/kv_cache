# 面向边缘设备 MoE 部署的层级专家驻留与负载结构

## 1. 需求背景：边缘设备上的 MoE 内存瓶颈

MoE 模型通过稀疏激活扩大参数容量：每个 token 只激活少量 expert，因此计算量不会随总参数规模线性增长。但在边缘设备或显存受限设备上，MoE 推理仍然面临一个核心瓶颈：虽然单个 token 只使用少量 expert，一条序列或一个 batch 中的不同 token 往往会共同访问大量不同 expert，导致系统仍然需要让大量 expert 权重常驻在 HBM 中。

如果所有 expert 都必须常驻，MoE 的总参数规模很快超过边缘设备的可用显存。一个直接思路是 expert swapping：把高频访问的 expert 常驻 HBM，把低频 expert 放在 CPU memory 或更低层存储中，需要时再换入。这样做的关键问题不是单纯能否换入换出，而是 expert 访问模式是否具有稳定、可预测的 hot/cold 结构。

第二阶段研究已经指出，普通细粒度 expert swapping 在当前硬件条件下仍然强烈受带宽限制。即使 expert 预测准确率达到 100%，在现实设置中，由于 PCIe 带宽限制与系统开销，expert 权重传输时间仍可能比每层计算时间高出约 $2$-$3\times$。因此，朴素的动态换入换出会引入约数倍量级的额外时延，难以完全隐藏在计算后面。

这意味着，边缘设备部署 MoE 不能只依赖“预测下一个 expert 并换入”的细粒度策略。系统需要一种更天生显存友好的 expert 激活结构：少量高频 expert 长期驻留，大量低频 expert 才作为动态 swapping 对象。换言之，边缘部署真正需要的不是所有 expert 负载完全均匀，而是一个可解释、可预测、可管理的专家负载结构。

## 2. 数据先验：真实世界本身是 Zipf / 二八分布

一个经典认识是，人类世界和语言数据天然呈现 Zipf / long-tail / 二八分布。网页访问、文件访问、数据库查询、推荐请求、词频分布、主题分布都不是均匀的。计算机系统长期以来一直在利用这种分布设计缓存和存储层次：高频对象常驻更快的存储层，低频对象放在更慢但容量更大的存储层。

语言模型面对的数据也应当有类似结构。数据中的 feature 不是均匀出现的，而是层次化且长尾的。某些 feature 几乎所有数据都会用到；某些 feature 只在某个领域或类别中高频出现；更细粒度的 feature 则只在长尾子类中出现。

例如，在数学文本中，不同层级的 feature 可以理解为：

```text
所有数学证明共享的 common feature:
  出现“证明”后，后文更可能进入推理步骤；
  证明结尾附近更可能出现“因此”“故”“证毕”等表达。

代数类数据的 group feature:
  群、环、域、同态、理想、方程、抽象符号更常出现；
  某些表达式后续更可能接代数变换或符号推导。

几何类数据的 group feature:
  角、边、圆、平行、垂直、三角形、角度符号更常出现；
  某些表达式后续更可能接图形关系或角度关系。

更细粒度的 leaf feature:
  代数内部还有群论、线性代数、交换代数等；
  几何内部还有欧氏几何、解析几何等。
```

这和简单的上下文规则是同一类现象：

```text
前文有“因为”  -> 下一个 token 出现“所以”的概率升高；
前文有“虽然”  -> 下一个 token 出现“但是”的概率升高。
```

这里的 feature 不是某个 token id 本身，而是上下文中存在的预测规则、语义条件或任务子结构。如果 expert 真的学习到 feature specialization，那么 expert access 也不应天然均匀，而应继承这些 feature 的层次化 Zipf 分布。

## 3. 结构定义：Hierarchical Common-Unique MoE

如果数据 feature 本身是层次化的，那么 MoE 的 expert 结构也不应只是一个 flat expert pool。更自然的结构是 Hierarchical Common-Unique MoE：每个层级都包含一个 common expert 和若干 unique experts，common expert 负责该层级所有样本共享的高频 feature，unique experts 负责该层级内部更细、更低频的差异。

![Hierarchical Common-Unique MoE](figures/hierarchical_common_unique_moe.svg)

图 1：一个两级 $4\times4$ Hierarchical Common-Unique MoE 示例。顶层包含 1 个 global common expert 和 4 个 unique expert groups；图中展开其中一个 group，展示其内部的 1 个 group common expert 和 4 个 group unique experts。global common expert 与被选中 group 的 common expert 构成高频常驻路径，group unique experts 对应长尾 feature，是主要的动态 swapping 对象。

形式化地，考虑某一 Transformer 层中的 token 表示 $h\in\mathbb{R}^d$。一个 $L$ 级 Hierarchical MoE 可以由一组层级 expert 组成。第 $\ell$ 级包含一个 common expert：

$$
f^{(\ell)}_{\mathrm{com}}:\mathbb{R}^d\rightarrow\mathbb{R}^d
$$

以及若干 unique experts：

$$
\{f^{(\ell)}_{j}:\mathbb{R}^d\rightarrow\mathbb{R}^d\}_{j=1}^{n_\ell}.
$$

第 $\ell$ 级 router 记为：

$$
r^{(\ell)}(h)=\operatorname{TopK}_{k_\ell}\left(\operatorname{softmax}(W^{(\ell)}_r h)\right),
$$

其中 $r^{(\ell)}(h)\subseteq\{1,\dots,n_\ell\}$ 是该层级被选择的 unique expert 集合，$k_\ell$ 是该层级的 top-$k$ 数量。该层级输出为：

$$
y^{(\ell)}(h)
=
f^{(\ell)}_{\mathrm{com}}(h)
+
\sum_{j\in r^{(\ell)}(h)}
\alpha^{(\ell)}_j(h) f^{(\ell)}_j(h),
$$

其中 $\alpha^{(\ell)}_j(h)$ 为 router 给出的归一化 routing weight。若采用多级串联结构，则 token 在第 $\ell$ 级的状态可以写为：

$$
h^{(0)} = h,\qquad
h^{(\ell+1)} = h^{(\ell)} + y^{(\ell)}(h^{(\ell)}).
$$

最终输出为：

$$
\operatorname{HMoE}(h)=h^{(L)}.
$$

对于推理系统，关键不是上述公式本身，而是不同 expert 的驻留策略。我们可以将 expert 参数集合分为：

$$
\Theta_{\mathrm{resident}}
=
\{ \theta(f^{(\ell)}_{\mathrm{com}}) \}_{\ell=1}^{L}
\cup
\Theta_{\mathrm{hot\ group}},
$$

以及：

$$
\Theta_{\mathrm{swap}}
=
\{ \theta(f^{(\ell)}_j): j\in \text{long-tail unique experts}\}.
$$

也就是说，global common expert 和高频 group common expert 常驻 HBM，低频 leaf unique experts 动态换入换出。这种结构把数据分布、模型结构和存储层次对齐：

```text
高频 common feature -> common experts -> 常驻 HBM
中频 group feature  -> group common experts -> 常驻 / 预取
低频 tail feature   -> unique experts -> 动态 swap
```

## 4. 为什么现有 MoE 没有显式学出这种负载结构：负载均衡损失的作用

现有 MoE 训练中常见 load balance loss，它的提出有合理工程动机。早期大规模 MoE 训练需要避免不同 GPU 负载差异过大，否则部分 GPU 吃不满，整体训练吞吐下降。同时，load balance loss 也可以缓解 router collapse、dead expert 等优化动力学问题。

但是，这种目标往往默认把 expert load imbalance 视为需要消除的问题。事实上，expert load imbalance 至少有两种来源：

```text
来源一：优化动力学失败导致的不健康不均衡。
例如 router collapse、dead expert、少数 expert 无意义吃掉大部分流量。

来源二：语言数据 feature 长尾分布导致的可解释不均衡。
例如高频 feature 对应的 expert 被频繁访问，低频 feature 对应的 expert 很少访问。
```

前者确实需要避免，后者则未必是坏事。对于边缘设备上的 expert swapping，数据分布诱导出的 hot/cold expert 访问结构正是可利用的系统局部性。传统 load balance loss 并不区分这两类不均衡，而是把它们一起抹平。它解决了优化失败导致的不均衡，也同时抹掉了数据本身 Zipf 分布带来的有用 hot/cold skew。

为了验证这一点，我们构造了一个 synthetic 实验，直接检验两个问题：

```text
数据 feature 的 Zipf 分布会自然诱导 expert 负载不均衡；
负载均衡 loss 会消灭这种本来可以用于 expert swapping 的有益不均衡。
```

实验数据由层级 token pattern 生成。首先生成一组底层 local slots，每个 local slot 是固定长度 token pattern；然后生成一组 high slots，每个 high slot 由若干 local slots 组合而成；最后按 Zipf 分布采样 high slots 并展开成 token sequence。这样，每个 token 都有明确的 ground-truth local slot id 和 high slot id，便于分析 attention 和 MoE routing 是否对齐这些 feature。

在 `synthetic_zipf_alpha = 1.1` 的设定下，数据本身的偏斜非常明显：top8 local slots 约占全部 token positions 的 46.37%，top8 high slots 约占 62.91%。这说明数据中确实存在强烈的高频 feature。

首先，attention 已经很好地捕捉到了 high slot 信息。模型在该合成任务上的 token accuracy 达到约 94.4%，说明模型确实学到了数据生成过程中的可预测结构，而不是只记住了局部 token 统计。进一步分析 attention pattern 可以看到，约 75% 的 attention mass 落在正确的当前 high-slot 相关位置内，说明 high-slot feature 已经成为模型组织上下文检索的主要结构。

剩余约 25% 的 attention mass 会落到不同 high slots 之间的 token 上。在当前数据生成范式下，不同 high slots 之间不存在真实的预测依赖，因此这部分跨 high-slot attention 可以视为噪声关系。干预实验进一步验证了这一点：把跨 high-slot 的 KV mask 掉、只保留 same higher-level unit 的 KV 后，推理 accuracy 从 94.43% 仅变为 94.36%，几乎没有下降。

| attention mask | loss | accuracy | visible KV |
|---|---:|---:|---:|
| full attention | 0.1979 | 94.43% | 100% |
| same higher-level unit only | 0.1991 | 94.36% | 32.73% |

这个结果说明，模型真正依赖的 retrieval bucket 正是 high slot / higher-level feature。换言之，模型已经学到了数据中的 feature 组合结构；跨 high-slot attention 被 mask 后不影响生成准确率，也说明剩余 25% 的跨 high-slot 关系并不是当前任务所需的有效预测信息。后续 expert 负载实验是在一个模型确实理解合成 feature 的前提下进行的。

其次，在没有 load balance loss 的 MoE 中，模型本身也表现出按照不均衡 feature 分布进行路由的倾向。top8 local slots 在中间层有约 77.8% 的流量进入 top2 experts：

```text
no load balance, layer 1:
  top8 local slots -> top1 expert coverage: 46.6%
  top8 local slots -> top2 expert coverage: 77.8%
  global expert load: 25.8% / 48.0% / 14.4% / 11.8%
```

这里的 top2 expert coverage 指的是：真正最高频的 top8 local slots 并没有均匀散到所有 expert，而是主要集中到了两个 hot experts 上，这两个 experts 合计解释了 77.8% 的 top8 local-slot traffic。从 expert swapping 的角度，这已经形成了明确的系统局部性：高频 feature traffic 被压到少数 hot experts 上，因此这些 experts 可以作为 HBM resident candidates。

然后我们加入经典 MoE load balance loss：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{LM}}
+
\lambda \mathcal{L}_{\mathrm{balance}}.
$$

测试 $\lambda=0.001,0.01,0.1$。结果显示，任意一个权重都足以把全局 expert load 几乎压成 25% / 25% / 25% / 25%。同时，top8 local slots 的 top2 expert 覆盖率从原来的约 70% 到 78% 降低到约 51% 到 55%，接近 4 个 expert 均匀分布下 top2 experts 理论覆盖 50% 的随机水平。也就是说，加了负载均衡 loss 后，原本集中到两个 hot experts 的高频 feature traffic 被强制打散，重新变成近似均匀访问。与此同时，LM loss 和 accuracy 基本没有变化。

代表性结果如下：

| setting | layer | global max/min load | top8 local top1 coverage | top8 local top2 coverage |
|---|---:|---:|---:|---:|
| no load balance | 1 | 48.0% / 11.8% | 46.6% | 77.8% |
| $\lambda=0.001$ | 1 | 25.1% / 24.9% | 25.9% | 51.4% |
| $\lambda=0.01$ | 1 | 25.1% / 24.8% | 26.3% | 51.7% |
| $\lambda=0.1$ | 1 | 25.4% / 24.6% | 26.3% | 52.1% |

这个结果的含义非常明确。没有负载均衡时，模型会把高频 feature 的大部分流量集中到少数 hot experts；加入负载均衡后，这种集中性被系统性抹平，top2 coverage 回到约 50%。因此，MoE 本身具有捕捉数据 feature 不均衡并形成 hot/cold expert load 的倾向；传统 load balance loss 会把这种数据驱动的不均衡和优化失败导致的不均衡一起抹平，从而破坏 expert swapping 所需的访问局部性。

## 5. 真实数据验证：Hierarchical MoE 的建模能力与系统收益

在上述数学建模和负载结构实验基础上，我们进一步在真实数据上验证 Hierarchical MoE。在相近参数预算和相近计算量下，Hierarchical MoE 保持接近普通 flat MoE 的建模能力，同时提供更适合推理系统的 expert residency 结构。

我们已经将实验从 synthetic setting 推进到真实语言数据与真实 benchmark 评测。当前 checkpoint 使用同一基础模型、同一训练步数和同一评测方案，对比普通 flat MoE 和 Hierarchical MoE 两种结构。两种模型处于约 $0.6$B 参数、$5$B token 训练 setting，Hierarchical MoE 在五个真实任务上的平均指标达到 $38.09\%$，普通 flat MoE 为 $37.85\%$，整体表现持平并略高。

| model | ARC-Easy | HellaSwag | PIQA | RACE | SIQA | Average |
|---|---:|---:|---:|---:|---:|---:|
| flat MoE | 32.37% | 26.29% | 55.82% | 41.49% | 33.27% | 37.85% |
| Hierarchical MoE | 32.66% | 25.92% | 55.93% | 41.99% | 33.93% | 38.09% |
| delta | +0.29 pp | -0.37 pp | +0.11 pp | +0.51 pp | +0.67 pp | +0.24 pp |

同时，真实数据 checkpoint 的 routing 统计显示，expert routing 已经学习到数据分布中的差异性，并且没有出现 expert collapse。flat MoE 的四个 experts 都被稳定使用，最终路由占比分别约为 $27.46\% / 31.47\% / 23.94\% / 17.13\%$；Hierarchical MoE 的四个 top-level groups 也都被稳定使用，最终 group routing 占比分别约为 $25.95\% / 25.78\% / 26.27\% / 22.00\%$。在 leaf unique expert 层面，所有 leaf experts 均有流量，单个 leaf 的全局占比约在 $3.56\%$ 到 $8.73\%$ 之间。这说明模型并不是机械地把所有 token 平均打散，也没有让少数 expert 吃掉全部流量，而是在真实数据上形成了有差异、可解释、且健康的 routing 结构。

这个结果说明，Hierarchical MoE 的 common-unique 结构没有以牺牲模型能力为代价换取系统友好性。相反，在真实任务评测中，Hierarchical MoE 用更少参数实现了与普通 flat MoE 基本一致的任务表现。这为后续系统侧优化提供了关键前提：我们可以在不显著损害模型质量的情况下，把 expert 组织成更适合 HBM resident 与 long-tail swapping 的层级结构。

从系统角度看，真实任务结果的意义在于确认模型侧可行性：Hierarchical MoE 能维持与 flat MoE 相当的建模能力，因此第 6 节中的 expert residency 和 swapping 成本分析可以建立在一个质量可比的模型结构上，而不是建立在牺牲模型效果的前提上。

## 6. Expert swapping 成本分析

Hierarchical MoE 的系统收益来自把常用参数和长尾参数分开。common experts 和高频 group experts 可以常驻 HBM，低频 leaf unique experts 才需要动态换入换出。

端侧部署场景下，这一收益应按逐 token decode 来理解。端侧设备通常 batch size 很小，甚至每次只有一个 token 在做前向，因此系统真正关心的是“当前 token 还需要从外存 load-on-demand 多少 expert 参数”，而不是大 batch 内一共 touched 了多少个不同 expert。在这种部署假设下，Hierarchical MoE 的 common/unique 分解直接降低了每个 token 的动态换入参数量。

先看理论模型。设 Transformer hidden size 为 $d$。在每一层中，attention 包含 $Q,K,V,O$ 四个线性矩阵，每个矩阵规模为 $d\rightarrow d$，因此 attention 参数规模约为：

$$
P_{\mathrm{attn}} \approx 4d^2.
$$

忽略 LayerNorm、router bias 等低阶项后，设 flat MoE 有 $N$ 个 experts，每个 expert 都是：

$$
d\rightarrow kd\rightarrow d,
$$

则单个 flat expert 参数规模约为：

$$
P_{\mathrm{expert}}^{\mathrm{flat}} \approx 2kd^2.
$$

flat MoE 单层总参数规模为：

$$
P_{\mathrm{flat}}
\approx
4d^2 + 2Nkd^2.
$$

端侧 load-on-demand 部署时，可以让 attention 和一个高频 expert 常驻，其余 experts 按需换入。此时 flat MoE 单层常驻参数规模为：

$$
R_{\mathrm{flat}}
\approx
4d^2 + 2kd^2.
$$

如果当前 token 被路由到非常驻 expert，则需要动态换入一个完整 flat expert：

$$
S_{\mathrm{flat}}
\approx
2kd^2.
$$

Hierarchical MoE 将 expert 拆成 common path 和 leaf unique path。设 global common expert 为：

$$
d\rightarrow k_c d\rightarrow d,
$$

每个 group common expert 为：

$$
d\rightarrow k_g d\rightarrow d,
$$

每个 leaf unique expert 为：

$$
d\rightarrow k_u d\rightarrow d.
$$

若共有 $G$ 个 groups，每个 group 内有 $M$ 个 leaf unique experts，则 H-MoE 单层总参数规模为：

$$
P_{\mathrm{hier}}
\approx
4d^2
+ 2k_c d^2
+ 2Gk_g d^2
+ 2GMk_u d^2.
$$

端侧部署时，attention、global common expert 和所有 group common experts 常驻 HBM，仅 leaf unique experts load-on-demand，因此 H-MoE 单层常驻参数规模为：

$$
R_{\mathrm{hier}}
\approx
4d^2
+ 2k_c d^2
+ 2Gk_g d^2.
$$

因此，flat MoE 和 H-MoE 的常驻参数比例分别为：

$$
\rho_{\mathrm{flat}}
=
\frac{R_{\mathrm{flat}}}{P_{\mathrm{flat}}}
=
\frac{4d^2 + 2kd^2}{4d^2 + 2Nkd^2}
=
\frac{4 + 2k}{4 + 2Nk},
$$

$$
\rho_{\mathrm{hier}}
=
\frac{R_{\mathrm{hier}}}{P_{\mathrm{hier}}}
=
\frac{4d^2 + 2k_c d^2 + 2Gk_g d^2}
{4d^2 + 2k_c d^2 + 2Gk_g d^2 + 2GMk_u d^2}
=
\frac{4 + 2k_c + 2Gk_g}
{4 + 2k_c + 2Gk_g + 2GMk_u}.
$$

对单个 token 来说，H-MoE 仍然会执行 global common expert 和被选中 group common expert，因此本地计算路径没有消失；但动态换入只发生在最小粒度 leaf unique expert 上：

$$
S_{\mathrm{hier}}
\approx
2k_u d^2.
$$

因此，H-MoE 相对 flat MoE 的逐 token swapping 参数量比例为：

$$
\eta_{\mathrm{swap}}
=
\frac{S_{\mathrm{hier}}}{S_{\mathrm{flat}}}
=
\frac{2k_u d^2}{2kd^2}
=
\frac{k_u}{k}.
$$

只要 leaf unique expert 的 hidden expansion $k_u$ 显著小于 flat expert 的 hidden expansion $k$，H-MoE 就可以在常驻更多 common computation 的同时，把逐 token 动态换入参数量压到原来的很小一部分。更重要的是，common expert 和 group common expert 的计算发生在 HBM resident 参数上，系统可以在这段本地计算期间预取或换入 leaf unique expert，从而把更小的传输开销隐藏在 computation 后面。

在当前真实任务 checkpoint 对应的模型结构上，我们只采用不含 vocab embedding / LM head 的 Transformer 主干口径，因为这个口径更直接反映 MoE 结构本身对端侧 load-on-demand 的影响。在 flat MoE 中，常驻 attention、router 和每层一个高频 expert 后，常驻参数比例为 $37.99\%$；在 Hierarchical MoE 中，常驻 attention、router、global common expert 和所有 group common experts 后，常驻参数比例为 $53.91\%$。

也就是说，H-MoE 将 Transformer 主干中的常驻参数比例从 $37.99\%$ 提升到 $53.91\%$，增加 $15.92$ 个百分点；与此同时，逐 token 的动态 expert swapping 参数量降低到 flat MoE 的约 $12.5\%$，也就是约 $1/8$。这说明当前 H-MoE 结构已经实现了本文希望的系统形态：用更高比例的 resident common computation 换取更小粒度的 dynamic unique expert load-on-demand，从而为端侧推理中隐藏 swapping 时延提供结构基础。

这一分析说明，Hierarchical MoE 不只是模型结构上的归纳偏置，也能直接转化为边缘设备 MoE 部署时的显存和带宽收益。其核心目标不是消除所有负载不均衡，而是让 expert 负载与真实数据的层级 Zipf feature 分布对齐：common experts 常驻 HBM，long-tail unique experts 动态 swap。
