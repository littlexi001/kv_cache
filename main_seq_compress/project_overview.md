# Hierarchical Language Modeling: 逻辑闭环

## 背景

现代大模型推理的核心瓶颈之一是 KV cache。它的规模近似为：

$$
O(L \cdot N \cdot d)
$$

其中 $L$ 是层数，$N$ 是上下文长度，$d$ 是 hidden dimension。长上下文场景下，主要压力来自 sequence length $N$。

KV cache 瓶颈最初 motivate 我们重新思考模型结构，但它不应当是这个工作的唯一目的。更根本的问题是：语言数据本身是否具有层次化、组合式的结构；如果有，语言模型是否也应当从设计上形成层次化的信息组织方式，而不是在所有层、所有位置都保存完整 token-level 表征。

我们的直觉是：人处理长文本时，并不是平铺地记住所有 token，而是建立目录、索引、摘要和可检索的层次化记忆。KV cache 变小应当是这种模型结构的 byproduct，而不是唯一目标。

## 第 1 轮：语言模型能否形成可压缩的层次化记忆

- **假设**：语言模型不必在所有层保留完整 token-level memory；中间层可以形成更稀疏、更抽象的 anchor memory。

- **验证状态**：支持。

- **当前结论**：KV sequence 维度存在可压缩性，说明完整 token-level memory 不是所有层的必要条件。

## 第 2 轮：这种记忆结构是否稳定

- **假设 1：可压缩性只来自温和 setting。**  
  **结论**：不成立。更激进压缩仍然可训练，并呈现清晰的压缩-能力 trade-off。

- **假设 2：训练时的稀疏访问不一定对应真实推理记忆。**  
  **结论**：不成立。anchor-only KV decode 与 full-KV decode 基本一致，说明模型确实适配了可丢弃的 memory layout。

- **假设 3：普通任务不足以证明层次化记忆真的有效。**  
  **结论**：成立。普通 benchmark 能力可以保持，但仍不能证明长程检索、精确复制和代码引用能力。

## 第 3 轮：层次化记忆是否真的承载语言结构

- **假设 1：anchor memory 真的融合了局部 token 信息。**  
  **验证状态**：待验证。

- **假设 2：这种结构在长程检索、精确复制、代码引用中仍然有效。**  
  **验证状态**：待验证。

- **假设 3：相比普通 sparse attention / sliding window，这种层次化 memory layout 有独立价值。**  
  **验证状态**：待验证。

## 当前定位

这个工作不应被理解为“为了省 KV cache 而设计 sparse attention”。更准确的定位是：

> 从语言数据的层次化、组合式结构出发，训练语言模型形成层次化、可检索、可丢弃的推理记忆；KV cache 压缩是这种结构自然带来的系统收益。

详细实验记录见 [`fdong/unet_transformer.md`](fdong/unet_transformer.md)。
