# 第一轮：标准 Attention/MoE 是否自然形成 feature bucket

## 假设

人类语言，以及我们用来替代它的 synthetic 数据，可以被看成 hierarchical / compositional sequential data。

如果这个假设成立，那么带 MoE 的标准 Transformer 可能自然暴露出这种 hierarchy：

```text
hierarchical data
-> attention 给 feature-related token 更高 score
-> MoE 把 same-feature token route 到同一个 expert
-> expert id 变成 feature bucket / index
```

第一轮实验直接测试这个强版本假设。

## 实验目的

这一轮的目的还不是证明 inverse KV，而是先回答一个 baseline 问题：

> 在具有明确 ground-truth feature structure 的数据上，标准 Attention 和标准 MoE 是否会自然形成我们想要的 feature bucket？

如果答案是肯定的，现有 MoE routing 可能已经可以作为 KV index。  
如果答案是否定的，feature-indexed routing 就需要显式的结构设计或训练设计。

## 合成数据

数据直接生成 token id 序列，不依赖自然语言文本。

最小结构单元是固定长度 slot。例如 `block_size = 4` 时，一个 local slot 是一个 4-token pattern。数据集初始化时会固定一个 slot pool，因此同一个 seed 下，不同 step / batch 使用的是同一套 slot pool。

更高层 hierarchy 由低层 unit 组合而成：

```text
layer 0: token -> local slot
layer 1: local slots -> higher-level unit
```

每个样本最终展开成 token 序列，并使用标准 causal language modeling 训练：

```text
input  = S[0:n-1]
target = S[1:n]
```

这样每个 token 都有 ground-truth 标签：

- local slot id；
- higher-level unit id；
- 两个 token 是否共享同一个 feature；
- MoE expert assignment 是否与 feature label 对齐。

## 模型

第一轮 baseline 使用小型 causal Transformer：

- 3 层 Transformer；
- 标准 causal self-attention；
- FFN 替换为标准 top-1 MoE；
- 4 个 unique experts；
- 不使用 common expert；
- hidden size 为 128；
- sequence length 为 128；
- local slot length 为 4。

这测试的是：在没有额外约束时，标准 MoE router 是否自然变成 feature router。

## 结果

模型学会了 synthetic sequence task。Next-token accuracy 达到约 90%+，说明任务本身可学，模型也确实捕捉到了有用结构。

Attention 对 ground-truth local structure 有明显敏感性：

```text
same-slot attention score > non-slot attention score
```

但是 same local slot 只拿到约 20% 的总 attention mass。也就是说，attention 使用了 slot 信息，但不是以“绝大部分 attention mass 都集中到 same-slot token”这种强形式组织计算。

标准 MoE 没有自然形成 feature bucket：

```text
same-slot token 没有稳定 route 到同一个 expert
same-expert token 的 attention 相似性没有显著更高
expert id 与 local slot / higher-level unit 的对齐很弱
```

## 结论

第一轮削弱了最初的强假设：

```text
hierarchical data
=> 标准 Attention 和标准 MoE 会自然暴露 hierarchy
=> expert id 可以直接作为 KV index
```

更合理的修正结论是：

```text
hierarchical data
=> 模型可以利用 hierarchical features
=> 但标准 Attention / 标准 MoE 不一定把它们暴露成干净、可索引的 routing bucket
```

重要的结构问题是，标准 MoE routing 发生在 attention 之后：

```text
hidden state -> attention -> MoE / FFN
```

因此，即使 MoE 中包含 feature 信息，它也不能直接指导当前层的 KV selection。至少需要把 routing 前置到 attention 之前，或者只把它用于组织后续层 / 后续 step 的 memory。
