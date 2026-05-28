# 面向端侧 MoE 部署的 Feature-Level Specialization 与层级专家换入换出

## 1. 二阶段结论：现有 MoE swapping 的两个核心瓶颈

MoE 模型通过稀疏激活扩大参数容量：每个 token 只激活少量 expert，因此计算量不会随总参数规模线性增长。但在端侧设备或显存受限设备上，MoE 推理的核心瓶颈不是 FLOPs，而是 expert 权重的驻留和换入换出。

第二阶段研究显示，直接对现有 flat MoE 做细粒度 expert swapping 会遇到两个同时存在的问题。

第一，跨 token 的 expert 预测准确率不足。当前 MoE router 通常基于当前 token 的完整 hidden representation 分发，其中包含大量 residual token identity、位置和局部表征信息。这样的分发方式容易被当前 token 主导，而不是稳定地对应某个可跨 token 预测的 feature。因此，即使某个 expert 在当前 token 被激活，也不容易可靠预测下一 token 会激活哪个 expert。

第二，即使 expert 预测达到 100% 准确，要搬运的参数量仍然过大。在现实硬件条件下，由于 PCIe 带宽限制和系统开销，完整 expert 权重传输时间仍可能比每层计算时间高出约 $2$-$3\times$。也就是说，朴素地预测下一个 expert 并换入，并不能自然把访存时延隐藏在计算后面。

我们认为，这两个挑战本质上都来自同一个原因：现有 MoE 模型没有实现 feature-level specialization。

```text
可预测性差：
  router 使用含 residual 的完整表征分发，routing 被 token-level identity 主导；
  expert id 没有稳定对应可跨 token 预测的 feature bucket。

搬运参数太多：
  不同 experts 之间共享的 common feature 没有被拆出来；
  expert 内部不同层次 feature 也没有被拆开；
  因此每次 load-on-demand 都必须搬运一个完整 expert。
```

因此，端侧 MoE 部署需要的不是简单提高 expert 预测器精度，而是改变 MoE 的 expert 组织方式：让高频、共享、可预测的 feature 对应常驻 computation，让低频、差异化 feature 对应更小粒度的 dynamic unique expert。

## 2. 数据先验：真实语言数据天然具有层级 Zipf feature

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

这和简单上下文规则是同一类现象：

```text
前文有“因为”  -> 下一个 token 出现“所以”的概率升高；
前文有“虽然”  -> 下一个 token 出现“但是”的概率升高。
```

这里的 feature 不是某个 token id 本身，而是上下文中存在的预测规则、语义条件或任务子结构。如果 expert 真的学习到 feature specialization，那么 expert access 也不应天然均匀，而应继承这些 feature 的层次化 Zipf 分布。

## 3. 结构方案：Hierarchical Common-Unique MoE

如果数据 feature 本身是层次化的，那么 MoE 的 expert 结构也不应只是一个 flat expert pool。更自然的结构是 Hierarchical Common-Unique MoE：每个层级都包含 common expert 和 unique experts。common expert 负责该层级所有样本共享的高频 feature，unique experts 负责该层级内部更细、更低频的差异。

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
\alpha^{(\ell)}_j(h) f^{(\ell)}_j(h).
$$

对于推理系统，关键不是公式本身，而是不同 expert 的驻留策略：

```text
高频 common feature -> common experts -> 常驻 HBM
中频 group feature  -> group common experts -> 常驻 / 预取
低频 tail feature   -> leaf unique experts -> 动态 swap
```

这直接对应二阶段发现的两个问题。对于可预测性，expert id 不再只是当前 token identity 的副产物，而是更接近 feature bucket。对于搬运量，系统不再每次搬运完整 expert，而是把 common 部分常驻，只按需换入更小的 leaf unique 部分。

## 4. Synthetic 证据：模型能学到 feature，负载均衡会抹掉有用不均衡

为了验证 feature 分布与 expert load 的关系，我们构造了层级 synthetic 实验。数据由层级 token pattern 生成：底层 local slots 是固定长度 token pattern；high slots 由若干 local slots 组合而成；最后按 Zipf 分布采样 high slots 并展开成 token sequence。这样，每个 token 都有明确的 ground-truth local slot id 和 high slot id，便于分析 attention 和 MoE routing 是否对齐这些 feature。

在 `synthetic_zipf_alpha = 1.1` 的设定下，数据本身的偏斜非常明显：top8 local slots 约占全部 token positions 的 $46.37\%$，top8 high slots 约占 $62.91\%$。这说明数据中确实存在强烈的高频 feature。

首先，attention 已经很好地捕捉到了 high slot 信息。模型在该合成任务上的 token accuracy 达到约 $94.4\%$，说明模型确实学到了数据生成过程中的可预测结构，而不是只记住局部 token 统计。进一步分析 attention pattern 可以看到，约 $75\%$ 的 attention mass 落在正确的当前 high-slot 相关位置内，说明 high-slot feature 已经成为模型组织上下文检索的主要结构。

跨 high-slot 的 KV mask 实验进一步验证了这一点：只保留 same higher-level unit 的 KV 后，推理 accuracy 从 $94.43\%$ 仅变为 $94.36\%$，几乎没有下降。

| attention mask | loss | accuracy | visible KV |
|---|---:|---:|---:|
| full attention | 0.1979 | 94.43% | 100% |
| same higher-level unit only | 0.1991 | 94.36% | 32.73% |

其次，在没有 load balance loss 的 MoE 中，模型本身表现出按照不均衡 feature 分布进行路由的倾向。top8 local slots 在中间层有约 $77.8\%$ 的流量进入 top2 experts：

```text
no load balance, layer 1:
  top8 local slots -> top1 expert coverage: 46.6%
  top8 local slots -> top2 expert coverage: 77.8%
  global expert load: 25.8% / 48.0% / 14.4% / 11.8%
```

这里的 top2 expert coverage 指的是：真正最高频的 top8 local slots 并没有均匀散到所有 expert，而是主要集中到了两个 hot experts 上。这说明 MoE 本身具有捕捉数据 feature 不均衡并形成 hot/cold expert load 的倾向。

然后加入经典 MoE load balance loss：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{LM}}
+
\lambda \mathcal{L}_{\mathrm{balance}}.
$$

测试 $\lambda=0.001,0.01,0.1$。结果显示，任意一个权重都足以把全局 expert load 几乎压成 $25\% / 25\% / 25\% / 25\%$。同时，top8 local slots 的 top2 expert 覆盖率从原来的约 $70\%$ 到 $78\%$ 降低到约 $51\%$ 到 $55\%$，接近 4 个 expert 均匀分布下 top2 experts 理论覆盖 $50\%$ 的随机水平。

| setting | layer | global max/min load | top8 local top1 coverage | top8 local top2 coverage |
|---|---:|---:|---:|---:|
| no load balance | 1 | 48.0% / 11.8% | 46.6% | 77.8% |
| $\lambda=0.001$ | 1 | 25.1% / 24.9% | 25.9% | 51.4% |
| $\lambda=0.01$ | 1 | 25.1% / 24.8% | 26.3% | 51.7% |
| $\lambda=0.1$ | 1 | 25.4% / 24.6% | 26.3% | 52.1% |

这个结果的含义非常明确：负载不均衡有两种来源。一种是 router collapse、dead expert 等优化动力学失败；另一种是语言数据 feature 长尾分布导致的可解释不均衡。传统 load balance loss 不区分这两者，会把数据驱动的有用 hot/cold locality 一起抹平，从而破坏 expert swapping 所需的访问局部性。

## 5. 真实数据验证：建模能力保持，routing 健康且有差异

在 synthetic 机制验证基础上，我们进一步在真实语言数据和真实 benchmark 上比较普通 flat MoE 与 Hierarchical MoE。当前 checkpoint 使用同一基础模型、同一训练步数和同一评测方案。两种模型处于约 $0.6$B 参数、$5$B token 训练 setting，Hierarchical MoE 在五个真实任务上的平均指标达到 $38.09\%$，普通 flat MoE 为 $37.85\%$，整体表现持平并略高。

| model | ARC-Easy | HellaSwag | PIQA | RACE | SIQA | Average |
|---|---:|---:|---:|---:|---:|---:|
| flat MoE | 32.37% | 26.29% | 55.82% | 41.49% | 33.27% | 37.85% |
| Hierarchical MoE | 32.66% | 25.92% | 55.93% | 41.99% | 33.93% | 38.09% |
| delta | +0.29 pp | -0.37 pp | +0.11 pp | +0.51 pp | +0.67 pp | +0.24 pp |

真实数据 checkpoint 的 routing 统计进一步显示，expert routing 已经学习到数据分布中的差异性，并且没有出现 expert collapse。flat MoE 的四个 experts 都被稳定使用，最终路由占比分别约为 $27.46\% / 31.47\% / 23.94\% / 17.13\%$；Hierarchical MoE 的四个 top-level groups 也都被稳定使用，最终 group routing 占比分别约为 $25.95\% / 25.78\% / 26.27\% / 22.00\%$。在 leaf unique expert 层面，所有 leaf experts 均有流量，单个 leaf 的全局占比约在 $3.56\%$ 到 $8.73\%$ 之间。

这说明 Hierarchical MoE 的 common-unique 结构没有以牺牲模型能力为代价换取系统友好性。模型保持了与 flat MoE 基本一致的真实任务表现，同时形成了有差异、可解释、且健康的 routing 结构。这为后续 expert residency 与 load-on-demand swapping 成本分析提供了模型侧前提。

## 6. Swapping 成本分析：常驻比例提升，动态换入降到约 1/8

端侧部署场景下，这一收益应按逐 token decode 来理解。端侧设备通常 batch size 很小，甚至每次只有一个 token 在做前向，因此系统真正关心的是“当前 token 还需要从外存 load-on-demand 多少 expert 参数”，而不是大 batch 内一共 touched 了多少个不同 expert。

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

在当前真实任务 checkpoint 对应的模型结构上，我们采用不含 vocab embedding / LM head 的 Transformer 主干口径，因为这个口径更直接反映 MoE 结构本身对端侧 load-on-demand 的影响。在 flat MoE 中，常驻 attention、router 和每层一个高频 expert 后，常驻参数比例为 $37.99\%$；在 Hierarchical MoE 中，常驻 attention、router、global common expert 和所有 group common experts 后，常驻参数比例为 $53.91\%$。

也就是说，H-MoE 将 Transformer 主干中的常驻参数比例从 $37.99\%$ 提升到 $53.91\%$，增加 $15.92$ 个百分点；与此同时，逐 token 的动态 expert swapping 参数量降低到 flat MoE 的约 $12.5\%$，也就是约 $1/8$。这正好对应本文的核心目标：用更高比例的 resident common computation 换取更小粒度的 dynamic unique expert load-on-demand，从而让 leaf expert 的传输更有机会隐藏在 common path computation 后面。

## 7. 结论

二阶段结果说明，现有 flat MoE swapping 的问题不是单纯预测器不够强，而是模型结构本身没有实现 feature-level specialization。router 使用含 residual 的完整 token 表征，导致 expert activation 难以跨 token 预测；expert 本身又把 common feature 和 unique feature 混在一起，导致每次换入都要搬运完整 expert。

Hierarchical Common-Unique MoE 针对这两个问题给出统一结构解法：common experts 承载高频共享 feature 并常驻 HBM，group common experts 承载中频 group feature，leaf unique experts 承载低频差异化 feature 并动态 swap。Synthetic 实验证明模型能够学到 feature 组合结构，且数据 feature 的 Zipf 分布会自然诱导有用的 expert hot/cold locality；真实数据实验进一步证明该结构保持了与 flat MoE 相当的模型能力，并形成健康 routing。

在端侧逐 token decode 的部署场景下，H-MoE 将 Transformer 主干常驻参数比例从 $37.99\%$ 提升到 $53.91\%$，同时将逐 token 动态 expert 换入量降低到 flat MoE 的约 $1/8$。这使得 expert swapping 从“搬运完整 expert 的带宽瓶颈”转变为“在 resident common computation 后面隐藏小粒度 leaf unique expert 传输”的系统设计问题。
