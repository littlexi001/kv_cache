# Inverse KV 实验笔记

这个文件是 inverse KV 实验记录的入口。详细内容按假设 / 实验轮次拆分到独立文件中。

## 核心问题

如果序列数据由 hierarchical / compositional feature 生成，模型能否学到 feature-based routing，使携带相关 feature 的 token 形成可用于 KV reverse indexing 的 memory bucket？

当前工作链条是：

```text
hierarchical / compositional data
-> Attention 捕捉 feature relation
-> MoE / gating 按 feature routing
-> routing bucket 可以作为 inverse KV index
```

## 文件

1. [第一轮：标准 Attention/MoE 是否自然形成 feature bucket](./inverse_kv_round1_notes.md)

   测试标准 Attention + 标准 MoE 在 synthetic hierarchy 数据上是否会自然形成 feature bucket。

2. [第二轮：Attention readout 与 MoE selectivity](./inverse_kv_round2_notes.md)

   测试 attention 是否已经包含有效 feature retrieval structure，以及 MoE 变体是否能把这个结构读出来。

3. [第三轮：Gate selectivity 与 inhibition 机制计划](./inverse_kv_round3_plan.md)

   当前 TODO。验证 MoE selectivity 不足是否来自 gate / expert 训练动力学，而不是 attention feature 缺失。

## 当前简短结论

Attention output 中已经存在很强的 ground-truth feature signal。local slot 几乎线性可读，higher-level unit 也显著线性可读。标准 MoE routing 仍然无法形成干净 expert bucket，因此下一步问题不是简单地让 attention 更集中，而是解释为什么 gate / expert 训练没有把可读 feature 转化成 selective routing。
