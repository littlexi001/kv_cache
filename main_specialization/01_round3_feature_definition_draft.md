# Round3 Feature Definition Draft: Parameter-Space Features

## 核心动机

前两轮对 feature 的定义主要来自数据生成过程或语言数据的先验结构：

1. Round1 把 feature 定义为 synthetic data 中显式构造的 local slot / higher-level slot；
2. Round2 把 feature 定义为由 next-token distribution 诱导出的状态等价类。

这些定义有助于构造可控实验，但它们本质上仍然是从数据侧出发定义 feature。导师提出的新方向是：**从模型自身的参数空间出发定义 feature**。

如果大语言模型真的学到了某种 feature，那么这些 feature 不一定首先表现为数据里的 token / slot / state label，而可能首先表现为模型参数矩阵中的方向、基底或谱结构。

## 参数矩阵视角下的 Feature

考虑模型中的一个线性参数矩阵 `W`。在 PyTorch 中，常见计算形式可以写成：

```text
Y = X W^T
```

对 `W` 做 SVD：

```text
W = U Sigma V^T
```

等价地：

```text
W^T = V Sigma U^T
```

这里具体把 `U` 还是 `V` 称为 feature basis 取决于矩阵方向约定，但核心不变：SVD 把线性映射拆成一组正交方向、对应强度和输出方向。

可以把这个过程理解成：

1. 输入 `X` 先在一组 feature key 上做查询；
2. 每个 feature key 得到一个 activation / scale；
3. 这个 scale 乘上对应奇异值，表示该 feature 在当前参数矩阵中的重要性；
4. 最终输出是对一组 value / output direction 的加权组合。

因此，线性层可以被重新理解成一种 attention-like computation：

```text
input query -> feature keys -> singular-value-scaled feature weights -> output directions
```

在这个视角下，feature 不是人为标注的 slot，也不是数据分布中的等价类，而是模型参数矩阵谱分解后形成的可查询方向。

## 需要分析的问题

### 1. 单条数据如何激活参数矩阵中的 feature

给定一条数据的 hidden state `x`，可以分析它在参数矩阵 feature basis 上的投影强度。

核心问题：

1. 一条数据会稠密地激活所有 feature，还是只激活少数 feature？
2. 不同 token / context 的 feature activation 是否稀疏？
3. activation 的排序是否与参数矩阵的奇异值排序一致？
4. 高频 feature 是否对应更大的奇异值，还是奇异值大小与数据 activation 分布无关？

这能回答：模型中的 feature 是否表现为少数强激活方向，还是每条数据都在所有方向上混合表达。

### 2. 每个 feature 在整体数据上的激活分布

反过来，对每个参数矩阵 feature direction，可以统计整个数据集在该方向上的查询强度分布。

核心问题：

1. 某些 feature 是否被大量数据频繁激活？
2. 某些 feature 是否只被少量数据激活？
3. feature activation 是否呈现 Zipf / long-tail 分布？
4. feature activation 的长尾程度是否和自然语言数据分布一致？
5. MoE expert 是否更容易对应高频 feature、低频 feature，还是某种 feature cluster？

这能回答：模型参数中的 feature 是否真的反映了语言数据中的 long-tail / compositional structure。

## 与真实数据和合成数据的连接

这一轮的核心研究方式应当是：

1. 先在真实数据训练出的模型上测量参数矩阵 feature 的统计性质；
2. 再通过控制 synthetic data 的性质，观察哪些数据因素会复现真实模型中的 feature 统计现象。

也就是说，synthetic data 不再只是为了验证“同 slot 是否分到同 expert”，而是为了回答：

```text
数据中的哪些性质，会导致模型参数矩阵形成类似真实模型的谱结构和 feature activation 分布？
```

可能需要控制的数据性质包括：

1. token frequency 是否服从 Zipf 分布；
2. feature frequency 是否服从 Zipf 分布；
3. 是否存在 same-input-different-output；
4. 是否存在 different-input-same-output；
5. feature 是否 compositional；
6. feature 是否 hierarchical；
7. 不同 feature 是否共享 token / prefix / output；
8. 任务是否必须依赖 long-context 才能预测。

## 对 Specialization 的新定义

在 Round3 视角下，specialization 可以被重新定义为：

```text
不同 expert 是否对应参数空间中的不同 feature directions / feature subspaces。
```

这比前两轮定义更接近模型内部机制：

1. Round1 问的是 expert 是否对齐 synthetic slot；
2. Round2 问的是 expert 是否对齐 next-token distribution 等价类；
3. Round3 问的是 expert 是否对齐模型参数矩阵或表征空间中真实形成的 feature basis。

因此，Round3 的 specialization 指标不应只看 token / slot / group purity，还应看：

1. expert 内 token 在参数矩阵 feature basis 上的 activation 是否相似；
2. 不同 expert 是否负责不同 SVD feature directions；
3. expert routing 是否能由少数 feature activation 解释；
4. 同一 expert 是否对应某些高频 / 低频 feature；
5. MoE 参数矩阵自身的 SVD feature 是否与 routing pattern 对齐。

## 当前直觉

如果导师的直觉成立，那么真实模型中的 feature 应当表现为：

1. 参数矩阵奇异值分布 sharp；
2. 数据在 feature basis 上的 activation 不是均匀的，而是稀疏或长尾的；
3. 某些 feature direction 被大量数据查询，某些 feature direction 只被少量数据查询；
4. MoE expert 的分发应当与这些 feature activation pattern 有关；
5. synthetic data 只有在具备某些真实语言性质时，才会复现类似的谱结构与 activation 分布。

这轮工作的核心不是先假设 slot / distributional label 就是 feature，而是反过来问：

```text
模型参数空间中自然形成的 feature 是什么？
什么样的数据性质会导致这些 feature 出现？
MoE expert 是否能按照这些 feature 形成 specialization？
```

