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

## 第 4 轮：K-cache 是否适合组织成图索引

现在的问题进一步具体化为：

> 能否把 flat K-cache 组织成 centered K-space 上的稀疏图，让 query 先访问 anchor / cluster center，再展开少量候选 K/V，从而降低 full qK score 的计算成本？

本轮离线实验对 Qwen3-0.6B 的 K-cache 做了三个观察。

### 4.1 K 相似性有可稀疏化结构

Raw K-K similarity 很高，但主要受 common direction 影响；因此真正用于建图的对象应当是 centered / residual K。

在 centered token-level cosine 下，top-k neighbor 仍有明显区分度：

```text
top-10: mean 0.5397, p50 0.5239, p95 0.7981
top-20: mean 0.4707, p50 0.4481, p95 0.7552
top-50: mean 0.3724, p50 0.3434, p95 0.6817
```

这说明 K-space 不是所有 token 都同等相关；强边和弱边之间有差异，支持从 complete weighted graph 中删掉低相似边，保留 sparse high-similarity graph。

### 4.2 高相似 K 边不只是局部边

如果 K similarity 只连接相邻 token，那么它更像局部平滑或 block compression，而不是知识图谱式 memory graph。实验显示 centered K 的高相似边有稳定比例跨越较远距离：

```text
token cos top-10:
  distance >=128: 24.9%
  distance >=256: 8.7%
  p95 distance: 397

token cos top-20:
  distance >=128: 30.7%
  distance >=256: 12.0%
  p95 distance: 462

token cos top-50:
  distance >=128: 40.2%
  distance >=256: 17.8%
  p95 distance: 561
```

Head-level 中一些 head 的非局部性更强：

```text
L06H3 top-10:
  distance >=128: 60.2%
  distance >=256: 34.8%

L15H1 top-10:
  distance >=128: 57.6%
  distance >=256: 35.6%

L02H6 top-10:
  distance >=128: 56.9%
  distance >=256: 32.0%
```

这支持继续研究 head-wise K graph，而不是只做局部 window / block 结构。

### 4.3 K 的 common direction 不决定 qK attention 选择性

实验还解释了一个关键现象：K 中存在巨大 common direction，导致 raw K-K similarity 很高，但 qK attention score 仍然有选择性。

令：

```text
k_i = c + r_i
```

则：

```text
q · k_i = q · c + q · r_i
```

其中 `q · c` 对同一个 query 的所有历史 token 都是同一个常数，因此在 softmax 中抵消：

```text
softmax(q · c + q · r_i) = softmax(q · r_i)
```

实验验证：

```text
mean cos(K, mean K) = 0.791
p50  cos(K, mean K) = 0.810
p95  cos(K, mean K) = 0.986

mean cos(q, mean K)     = 0.108
p50  cos(q, mean K)     = 0.062
mean abs cos(q, mean K) = 0.122

raw score std      = centered score std
std ratio          = 1.0000
score correlation  = 1.0000
top-10 overlap     = 1.0000
attention JS       ~= 0
```

因此，K graph 应当建立在 centered / residual K geometry 上，而不是 raw K geometry 上。Raw common direction 会让图看起来虚假地稠密，但它对 attention 的 token 间选择性几乎没有贡献。

本轮结论是：

> 当前模型的 K-cache 具备建图的初步好性质：centered K 空间中强弱边有区分度，高相似边包含非局部长距离连接，部分 head 出现 hub / anchor 候选；并且 centered K 是 attention-relevant 的 address geometry。

但还没有证明：

> 这些 graph anchors / neighborhoods 能否召回真实 qK attention mass。

下一步关键实验应当是：

```text
centered K graph candidates
-> compare against full qK attention
-> measure attention mass recall / top-token recall / candidate budget
```

## 当前定位

这个工作不应被理解为“为了省 KV cache 而设计 sparse attention”。更准确的定位是：

> 从语言数据的层次化、组合式结构出发，训练语言模型形成层次化、可检索、可丢弃的推理记忆；KV cache 压缩是这种结构自然带来的系统收益。

详细实验记录见 [`fdong/unet_transformer.md`](fdong/unet_transformer.md)。
