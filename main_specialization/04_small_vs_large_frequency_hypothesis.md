# 小模型到底比大模型差在哪：宽度、频率与长尾表征干扰

## 0. 当前要回答的问题

老板的问题是：

> 小模型到底比大模型差在哪？

这个问题不能只回答“参数少”。参数量只是总量口径，不能说明能力差异到底来自哪里。我们更关心的是：模型变大时，哪些结构维度在变，哪些能力因此变强。

因此第一步要先定义这里的“大”和“小”。

## 1. 什么叫大模型，什么叫小模型

在这个问题里，我们暂时不把“大/小”定义成单纯的参数量，而定义成：

> 表征空间、并行子空间和可分解特征容量的大小。

具体到 Transformer，可以看几个结构量：

1. hidden size，也就是每个 token 的表征维度；
2. FFN intermediate size，也就是每层非线性变换的宽度；
3. attention heads / KV heads / head dim，也就是注意力子空间的数量和维度；
4. MoE expert 数量与 expert 宽度，也就是条件激活的参数子空间。

这个定义的好处是，它把问题从“参数多不多”转成：

> 模型有没有足够多、足够干净的表征方向去同时承载不同类型、不同频率、不同粒度的信息？

## 2. 现象：工业界 scale up 往往显著扩宽

我们观察到，工业界在 scale up Transformer 时，常见做法不是只加深，而是显著扩宽：

- hidden size 变大；
- FFN intermediate size 变大；
- attention head / head dim 变大；
- MoE 中 expert 数量或 expert 宽度变大。

这引出第一个问题：

> 为什么 scale up 时扩宽这么重要？为什么不是简单加深就够？

一个直觉回答是：扩宽提高了表征维度，表征空间里可以放下更多可区分的信息状态。

## 3. 第一版解释：宽度带来更大的信息表达空间

如果一个 token hidden state 是 $d$ 维向量，那么 $d$ 越大，模型可以使用的方向、子空间、组合特征就越多。从信息论直觉上看，更高维空间似乎能表达更多状态。

这给出一个朴素解释：

> 大模型更强，是因为它有更大的表征空间，可以编码更多信息。

但这个解释马上遇到一个反问。

## 4. 新问题：1024 维已经很大，为什么还要 8192 维？

如果把维度想象成离散编码容量，那么 $2^{1024}$ 已经是极大的数。直觉上，它甚至大到足以“编码整个世界”的大量状态。

所以真正的问题不是：

> 表征空间理论上能不能装下信息？

而是：

> 在真实训练、真实数据分布、有限优化、有限样本和相似性度量下，这些信息能不能被稳定、可分、可泛化、可检索地放在表征空间里？

也就是说，宽度的关键可能不是抽象容量，而是表征的可用性。

1024 维理论上可以表达很多东西，但如果很多重要信息都挤在同一批高能量方向附近，或者低频信息只能落在弱方向、噪声方向、混叠方向里，那么这些信息虽然“存在”，却不容易被下游层读出来，也不容易通过 L2 / cosine similarity 细粒度地区分。

## 5. 新假设：频率不均衡导致高频信息主导表征空间

真实语言和真实任务数据不是均匀分布的。它们有明显的 Zipf / long-tail 结构：

- 高频 token / pattern / feature 出现次数多；
- 高频 feature 产生更多梯度更新；
- 高频 feature 更容易占据 hidden state 的大方差方向；
- 表征矩阵的 top singular directions 可能主要编码高频因素；
- L2 / cosine similarity 会被这些大能量方向主导；
- 低频 feature 即使被模型学到，也可能被压在低能量、低信噪比、难分辨的方向里。

因此我们提出更具体的假设：

> 小模型的一个关键短板不是“完全没有能力记住低频信息”，而是在频率高度不均衡的数据中，有限宽度表征空间会被高频信息占据和污染，使低频信息难以形成足够独立、足够高信噪比的表征方向。

换句话说：

> 大模型的宽度优势，可能部分来自降低高频与长尾信息之间的表征干扰。

## 6. 为什么频率会产生影响

频率可能通过三条路径影响学习。

第一，优化路径。高频样本出现次数多，因此它们贡献更多梯度。模型更早、更稳定地为高频模式分配参数和表征方向。

第二，谱结构路径。如果某些高频 feature 在大量样本中共同出现，它们会成为 hidden states 中的主方差来源。SVD/PCA 看到的 top directions 更可能对应这些高频因素。

第三，相似性路径。常用 similarity metric 如 cosine / L2 会受大能量方向影响。如果高频方向占主导，那么两个样本即使低频属性不同，只要高频属性相同，也可能在表征空间里非常接近。这样低频差异就被 similarity metric 淹没。

这给出第二个研究问题：

> 低频学得差，到底是不是因为频率不均衡导致的表征干扰？

## 7. 要验证的核心预测

短期内我们先不试图完整证明“奇异值污染机制”，而是先验证三个较弱但关键的预测。

预测 1：

> 在均匀数据中，小宽度 Transformer 对不同 feature 的学习差异不大；在 Zipf / long-tail 数据中，小宽度 Transformer 对高频 feature 学得更好，对低频 feature 学得更差。

预测 2：

> 如果把 hidden size 从 64 / 96 增大到 128 / 192 / 256，低频 feature 的 loss / accuracy 应该改善，且改善幅度应高于高频 feature。

预测 3：

> 如果 somehow 把高频和低频 feature 分到不同专家或不同子空间，低频 feature 的学习会变好。

其中预测 1 和 2 是一小时内最适合先交的初步结果。预测 3 和表征谱分析更适合作为后续机制验证。

## 8. 现有数据逻辑能否复用

当前 workspace 没有找到 `.ts/.tsx` 数据生成文件，但已有 Python 数据生成逻辑可以直接复用。

最相关的是：

```text
fdong/scripts/utils/data_utils.py
```

其中 `HierarchicalPatternData` 已经支持：

- 层级 synthetic pattern；
- `sampling_distribution = uniform / zipf`；
- `zipf_alpha` 控制频率偏斜程度；
- 每个 token 返回 metadata；
- metadata 中包含 local slot id 和 higher-level unit id。

它的数据过程是：

1. 先生成 layer 0 local units，每个 local unit 是固定长度 token pattern；
2. 再生成 higher-level units，每个 high unit 由若干 local units 组合而成；
3. 训练序列通过采样 top-level high unit 并展开得到；
4. 如果采样分布是 Zipf，高频 high unit 会大量出现，低频 high unit 很少出现。

这个数据非常适合做第一版实验，因为我们可以明确知道每个 token 属于哪个 local slot / high slot，从而按频率桶统计 loss 和 accuracy。

## 9. 已有实验的提醒

已有 `H0529a_zipfian_frequency_shortcut` 实验显示：如果只是 token 频率 Zipf，但 token 条件下的 context / sense 仍然 balanced，那么 tail failure 不一定明显。

这说明这次不能只制造“token 出现频率不同”。更好的设置是：

> 让真正需要学习的 feature / high-level unit 本身呈现长尾分布。

`HierarchicalPatternData` 的 top-level unit Zipf 采样正好更接近这个目标。

## 10. 一小时内建议先做的实验

### 10.1 数据设置

第一版用两个数据条件：

```text
uniform:
  sampling_distribution = uniform

zipf:
  sampling_distribution = zipf
  zipf_alpha = 1.1 或 1.3
```

建议先用：

```text
seq_len = 128
block_size = 4
num_hierarchy_layers = 2
content_token_count = 256 或 512
num_units_per_layer = 64
synthetic_num_samples = 200000
```

解释：

- `block_size = 4` 保持任务容易，训练很快能看到 loss 下降；
- `num_hierarchy_layers = 2` 保留 local / high 两级 feature；
- `num_units_per_layer = 64` 让 tail unit 有足够多的类别；
- `zipf_alpha = 1.1` 是温和长尾，`1.3` 更容易打出高低频差异。

### 10.2 模型设置

先用 dense 小 Transformer，不上 MoE：

```text
num_hidden_layers = 2
hidden_size = 64, 96, 128
intermediate_size = 2 * hidden_size
num_attention_heads = 4
num_key_value_heads = 2
head_dim = hidden_size / 4
seq_len = 128
```

如果时间很紧，先跑：

```text
hidden_size = 64
conditions = uniform, zipf_alpha=1.3
steps = 1000 或 2000
```

如果还有时间，再补：

```text
hidden_size = 96 / 128
```

### 10.3 初步指标

最先交三个指标就够：

1. overall loss / accuracy；
2. high-frequency unit bucket loss / accuracy；
3. low-frequency unit bucket loss / accuracy。

bucket 可以按 high unit 的真实采样权重或 empirical count 划分：

```text
head: top 20% high units
middle: middle 40% high units
tail: bottom 40% high units
```

如果只用现有 `evaluate_capacity_boundary.py`，可以先交：

- overall；
- inside local；
- local boundary；
- high boundary。

但这还不是最理想的 head/tail 频率分析。更贴近问题的评测需要在该脚本上加一个 high-unit frequency bucket 统计。

## 11. 第一版结果应该如何判断

如果看到：

```text
uniform:
  head/mid/tail 差异小

zipf:
  head loss 明显低于 tail loss
  tail accuracy 明显低于 head accuracy

zipf + wider hidden:
  tail loss 下降幅度比 head 更大
```

那么可以先向老板报告：

> 初步结果支持“频率不均衡会让小宽度模型优先学高频 feature，低频 feature 学得更差；增加宽度可能缓解低频 feature 的学习困难”。

但还不能说：

> 已经证明高频信息通过大奇异值方向污染了 similarity metric。

这个机制性结论后面需要 SVD/PCA、cosine pair test、linear probe 或 whitening ablation 来补。

## 12. MoE 版本是否现在做

一小时内不建议把 MoE 作为第一主结果，因为 MoE 变量太多：

- 参数量会变；
- active parameter 会变；
- router 是否学到正确分工不稳定；
- MoE 改善可能来自参数更多，而不是高低频分离。

如果要做一个最小 MoE sanity check，建议只做 oracle / ground-truth routing：

```text
use_moe = true
moe_num_unique_experts = 4
moe_num_experts_per_tok = 1
ground_truth_routing_strategy = frequency_balanced 或 hash
ground_truth_routing_feature_layer = 1
```

这个设置的含义是：先人为把 high-level unit 分到专家里，看“分子空间”是否能缓解 tail loss。它不是最终自然 MoE 结论，但可以作为机制方向的 sanity check。

## 13. 推荐执行顺序

第一小时：

1. 跑 dense h64 的 uniform vs zipf；
2. 按 high-unit frequency bucket 评测 loss / accuracy；
3. 如果 zipf 下 tail 更差，再补 h96 或 h128；
4. 给老板交一张表：condition × hidden size × head/mid/tail loss。

第二阶段：

1. 加 h192 / h256，看 widening 是否持续改善 tail；
2. 加 oracle MoE / frequency-balanced routing；
3. 加 SVD/PCA，检查 top singular directions 是否更对齐高频 feature；
4. 加 cosine similarity pair test，检查低频差异是否被高频因素淹没。

## 14. 当前最稳的表述

现在最稳的研究口径是：

> 我们怀疑小模型相对大模型的一个重要短板，不只是信息容量不足，而是有限宽度表征空间在长尾数据分布下更容易被高频 feature 主导。高频 feature 由于出现次数多，会优先占据高能量表征方向；低频 feature 虽然理论上可以被编码，却更容易落在弱方向或混叠方向里，从而在 loss、accuracy 和 similarity 上表现更差。增加宽度或引入分专家结构，可能通过提供更多独立子空间来缓解这种高低频表征干扰。

一小时内的实验目标不是完整证明机制，而是先验证：

> 频率不均衡是否确实让低频 feature 在小宽度模型中更难学，以及增宽是否优先改善低频 feature。

