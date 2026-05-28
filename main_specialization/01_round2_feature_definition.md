# Round2 Feature Definition: Synthetic Data with Reused Tokens

第二轮 synthetic 数据不是简单地“让 token 随机重复”，而是从更根本的语言建模出发重新定义 feature。我们把语言看成一个概率状态机：当前位置状态 `s_t` 诱导出未来 token 的条件分布 `P(x_{t+1} | s_t)`。

因此，最细粒度的 functional feature 不是 token id，也不是某个唯一 next token，而是由未来分布诱导出的等价类。

在这个定义下，feature 可以有不同粒度。最细粒度 feature 是完整的 next-token distribution；更粗粒度 feature 可以是这个分布的某个 projection，例如 `P(next=A)`、某类 token group 的概率，或者未来分布在某个子空间上的坐标。复杂状态可以看成多个 distributional features 的组合。

Round1 的 clean synthetic 数据相比真实语言缺少两类关键现象：

1. **same-input-different-output：** 同一个输入状态可以对应多个可能输出 token，本质上是同一个条件分布的多次采样；
2. **different-input-same-output：** 不同输入状态可以共享相同或相近的 output-side distributional feature。

例如，`slot_size=4` 时，slot 被写成：

```text
input = (前三个 token)
output = (第四个 token)
```

则：

```text
ABCX
ABCY
ABCZ
```

属于 same-input-different-output。它表示同一个状态 `ABC` 的 next-token distribution 可以产生 `X/Y/Z`。这些样本不应被理解成三个互斥的 deterministic label，而应被看成同一个状态分布 feature 的不同观测。

而：

```text
ABCX
DEFX
GHIX
```

属于 different-input-same-output。它表示不同输入状态共享 `P(next=X)` 高这一 output-side projected feature；如果它们完整 next-token distribution 相同，则它们就是完整的 distributional equivalence class。

因此，Round2 更关心的是：当 token identity、input prefix、output token 与 distributional feature 不再一一对应时，MoE gate 到底会按什么分发。理想 specialization 不应只按 surface token id 分发，而应与 future-distribution feature 的相似性单调相关。

这个设定也自然连接到 real corpus。对真实语料而言，我们通常无法提前知道某个 token position 的 ground-truth feature label，因此可以用 downstream behavior 来定义 feature：如果两个 token position 的 next-token logits 分布相似，说明模型认为它们应当被映射到相近的新状态，因此它们可以被视为具有相似 feature。

更形式化地说，对每个 token position，可以取模型在该位置的 next-token logits 或概率分布作为语义状态表示。如果两个位置的预测分布接近，它们应当具有相近的 downstream behavior；理想的 expert specialization 应当让这些位置更倾向于进入相同或相近 expert bucket。

因此，Round1 / Round2 synthetic 与 real corpus 的 feature 定义并不是两套互不相干的定义。它们的共同核心是：**feature 不只是 token id，而是决定模型未来分布的可复用结构。** Synthetic 数据给出可控的 ground truth，real corpus 则需要通过 next-token logits、representation similarity 或 attention retrieval pattern 等 proxy 来近似这一结构。

Round2 的完整建模与实验记录见 [Round6 文档](../fdong/inverse_kv_round6_plan.md)。
