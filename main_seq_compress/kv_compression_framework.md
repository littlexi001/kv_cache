# CSA 启发下的 KV Cache 压缩问题框架

Date: 2026-05-27

## 1. 当前问题意识：从 memory retrieval 到 KV 几何结构

我们现在讨论的不是一个已经收敛的具体结构，而是一个更高层次的问题框架：

> 如果 attention 本质上可以被看成 memory retrieval，那么 KV cache 压缩就不只是“把历史 token 变小”，而是“如何把历史记忆组织成可索引、可选择、可扩展的检索系统”。

这个问题意识现在又往前推进了一步：在设计具体 compressor / indexer 之前，我们需要先理解真实 LLM 的 KV cache 随着 `seq_len` 增长到底呈现什么数学结构。也就是说，不能一开始就假设“平均池化”“低秩压缩”“top-k block selection”一定合理，而应先把 K/V cache 当成随 prefix 增长的高维点云，观察它是否低有效维、是否各向异性、是否局部平滑、是否形成 block、主子空间是否稳定、新 token 是否带来 residual 信息。

我们对 Qwen3-0.6B 做了一轮 prefix-growth KV geometry 诊断：固定一条长文本，观察 prefix 从 `512` 增长到 `12000` token 时，每层每头 K/V cache 的高维点云结构如何变化。阶段性结论记录在：

```text
fdong_seq_compress/qwen3_kv_cache_geometry_findings.md
```

这轮实验把 KV cache compression 的方向进一步收窄为：

> K-cache 更像可压缩、可索引的 address space；V-cache 更像需要高保真保留的 content space。合理路线不是平均压缩 K/V，而是 K 侧索引化，V 侧高保真读取。

| 性质 | K-cache | V-cache |
| --- | --- | --- |
| 有效维度 | 随 sequence length sublinear 增长，远慢于 token 数增长 | 也随 sequence length sublinear 增长，但有效维度整体比 K 更高 |
| 各向异性 | 强，存在明显 common direction / cone effect | 相比 K 弱得多，整体更分散 |
| 去均值后的相似性 | centering 后 token-token 平均相似性显著下降，说明 raw similarity 受公共方向影响很大 | centering 后平均相似性也接近 0，但 raw similarity 本来就不高 |
| 局部平滑性 | 强，相邻 token 的 K 向量高度相似，更像一条平滑高维轨迹 | 弱，相邻 token 的 V 向量差异明显更大 |
| 小 block 结构 | 支持小尺度连续 block，尤其是 4/8/16 token block | 不支持简单连续 block average，即使很小 block 内部也较分散 |
| 主子空间稳定性 | 主子空间随 prefix 增长逐渐稳定，但弱方向仍会旋转 | 主子空间更稳定，长 prefix 下 dominant subspace 几乎不再明显变化 |
| 新 token novelty | 新 token 对已有 top subspace 仍有明显 residual，不能被很小固定 basis 完全解释 | 也有明显 residual，且内容侧 residual 不应被忽略 |
| 层间差异 | 浅层 K 尤其各向异性强，后层 K 的公共方向相对减弱 | 后层 V 的有效维度更高，内容性更强 |
| 适合承担的角色 | 更像 address / index / routing space | 更像 content / evidence / information payload |
| 对压缩的直接启发 | 适合研究去 common direction 后的小 block index、delta、change point、routing summary | 更适合保留 exact residual、multi-slot content，或在 K index 命中后按需 gather，不适合直接平均压缩 |

这与 DeepSeek MCA / CSA 和经典搜索系统的 index-content 分离非常接近：

```text
search engine:
  compressed index / inverted index / vector index
  -> candidate documents
  -> read high-fidelity document content

KV cache:
  compressed K-index / block summary
  -> candidate token blocks
  -> read high-fidelity V content
```

因此，当前最值得推进的方向可以命名为：

```text
K-indexed, V-faithful KV cache compression
```

也就是：

```text
centered / whitened K block summary -> candidate block selection
selected blocks -> exact V / residual V / multi-slot content read
```

DeepSeek CSA 给出的重要启发在于，它没有把压缩理解成固定 pooling 或推理后处理，而是把 compression 和 selection 都做成模型结构的一部分：

- 先用可学习矩阵从 hidden states 生成 compressed KV content；
- 再用可学习权重把局部 token block 压成 compressed KV；
- 同时构造 lightweight indexer key；
- 对每个 query 先做 query-to-block scoring；
- 只把 top-k compressed blocks 送进真正的 attention。

这件事的关键不是某个公式本身，而是它把 attention 拆成了两层：

```text
index layer:   先判断哪些历史 memory blocks 值得看
content layer: 再对被选中的 compressed KV 做真正 attention
```

这和搜索、推荐、数据库、向量检索系统中的典型两阶段结构是一致的：

```text
cheap candidate recall -> expensive exact ranking / reading
```

因此，CSA 对我们的意义不是“我们也做一个一模一样的 block compressor”，而是提醒我们重新定义 sequence compression 的研究对象。

## 2. 从 sequence compression 到 memory retrieval

我们之前在 sequence compression 里探索过 U-Net Transformer / hierarchical memory 的方向。核心假设是：

> 语言表示不一定需要在所有层保持 token-level resolution；深层表示可能天然对应更少、更抽象、更可组合的 semantic units。

这个方向试图从模型内部结构上减少有效 sequence length：

```text
token sequence
-> local phrase / span units
-> entity / event / code object units
-> global task state
```

现在看，CSA 与这个方向并不矛盾，而是提供了一个更工程化、更直接的 realization：

```text
local token block
-> learned compressed memory entry
-> indexer key
-> query-time sparse retrieval
```

二者的共同点是：

- 都承认 full token-level KV 不是每一层、每一步都必须全量访问；
- 都需要模型学习“哪些信息应该被压缩进高层 memory”；
- 都需要模型学习“当前 query 应该访问哪些 memory units”；
- 都必须保留必要的 token-level 精度路径，否则精确复制、代码符号、数字、引用会崩。

差异在于：

- U-Net / hierarchical memory 更像是在改造模型的内部表征层级；
- CSA 更像是在当前 Transformer 框架内显式加入 learned compressor 和 learned indexer；
- 搜索系统视角则进一步把 KV cache 看成一个动态更新、分层存储、按需访问的 memory database。

所以我们现在的更清晰表述应当是：

> Sequence compression 的本质不是压缩 sequence 本身，而是把 flat token memory 改造成 multi-resolution indexed memory。

## 3. Attention KV Cache 的特殊性

如果要从经典搜索/推荐领域借工具，必须先认识到 LLM KV cache 不是普通文档库，也不是普通向量数据库。它至少有以下特殊性质。

### 3.1 KV 是激活态记忆，不是静态文本

KV cache 存的是每层、每头的内部激活：

```text
K_l,h, V_l,h
```

它不是原始 token，也不是 sentence embedding。不同层、不同 head 的 KV 语义不同：

- 浅层可能更偏 lexical / local syntax；
- 中层可能承载实体、指代、局部结构；
- 深层可能更接近任务状态、推理路径、答案相关证据。

因此，KV indexing 不能简单等价于“给文本 chunk 做 embedding search”。它索引的是模型内部为了下一步预测而形成的可用记忆。

### 3.2 Query 是高频、低延迟、逐 token 变化的

网页搜索可以接受毫秒到百毫秒级检索，LLM decode 中每个 token 都要检索：

```text
q_t -> retrieve historical K/V -> produce y_t
```

这意味着 indexer 必须极轻量，且最好能和 attention kernel、KV page 管理、prefetch 机制耦合。任何复杂的外部检索结构都必须证明它的额外开销小于减少 KV 访问带来的收益。

### 3.3 KV 是动态增长的在线索引

外部知识库可以离线建索引，但 KV cache 在推理时随 token 增长：

```text
prefix length N -> N + 1 -> N + 2
```

所以它更像一个 streaming index：

- 每步插入新 memory；
- 历史 memory 热度不断变化；
- block 可能从 hot 变 warm / cold；
- 局部上下文、系统提示、用户请求、工具返回内容都可能改变访问分布。

### 3.4 Attention 既是 retrieval，也是 soft computation

传统检索只需要找出相关文档，但 attention 的输出是：

```text
sum_i softmax(q · k_i) v_i
```

它既有 candidate selection 的性质，也有连续加权计算的性质。压缩或稀疏化不能只看 top-k key recall，还要看 value aggregation 后对 logits / loss / behavior 的影响。

因此，评估指标不能只用“找没找到相关块”，还要包含：

- next-token loss / CE delta；
- 长程证据任务正确率；
- attention mass recall；
- selected KV 对最终 logits 的贡献；
- 精确复制、代码引用、数字一致性。

### 3.5 Qwen3 KV 几何诊断带来的更新

在 `fdong_seq_compress` 中，我们对 Qwen3-0.6B 做了一轮 prefix-growth KV geometry 诊断：固定一条长文本，观察 prefix 从 `512` 增长到 `12000` token 时，每层每头 K/V cache 的高维点云结构如何变化。阶段性结论记录在：

```text
fdong_seq_compress/qwen3_kv_cache_geometry_findings.md
```

本轮最重要的观察是：K-cache 和 V-cache 的数学结构并不对称。

| 性质 | K-cache | V-cache |
| --- | --- | --- |
| 有效维度 | 随 sequence length sublinear 增长，远慢于 token 数增长 | 也随 sequence length sublinear 增长，但有效维度整体比 K 更高 |
| 各向异性 | 强，存在明显 common direction / cone effect | 相比 K 弱得多，整体更分散 |
| 去均值后的相似性 | centering 后 token-token 平均相似性显著下降，说明 raw similarity 受公共方向影响很大 | centering 后平均相似性也接近 0，但 raw similarity 本来就不高 |
| 局部平滑性 | 强，相邻 token 的 K 向量高度相似，更像一条平滑高维轨迹 | 弱，相邻 token 的 V 向量差异明显更大 |
| 小 block 结构 | 支持小尺度连续 block，尤其是 4/8/16 token block | 不支持简单连续 block average，即使很小 block 内部也较分散 |
| 主子空间稳定性 | 主子空间随 prefix 增长逐渐稳定，但弱方向仍会旋转 | 主子空间更稳定，长 prefix 下 dominant subspace 几乎不再明显变化 |
| 新 token novelty | 新 token 对已有 top subspace 仍有明显 residual，不能被很小固定 basis 完全解释 | 也有明显 residual，且内容侧 residual 不应被忽略 |
| 层间差异 | 浅层 K 尤其各向异性强，后层 K 的公共方向相对减弱 | 后层 V 的有效维度更高，内容性更强 |
| 适合承担的角色 | 更像 address / index / routing space | 更像 content / evidence / information payload |
| 对压缩的直接启发 | 适合研究去 common direction 后的小 block index、delta、change point、routing summary | 更适合保留 exact residual、multi-slot content，或在 K index 命中后按需 gather，不适合直接平均压缩 |

这些性质对压缩 `seq_len` 维度的启发是：

第一，压缩重点应当从“每个 token 的 K/V 向量怎么一起变小”转向“如何少访问历史 token / block”。KV 的有效维度随 sequence length 增长很慢，说明长序列不是在不断线性创造新几何方向；但 new-token residual 仍然明显，说明简单把 head dimension 压到很小 rank 不足以保留全部信息。因此，主收益更可能来自减少被访问的 sequence positions，而不是只压缩每个 position 的 head_dim。

第二，`seq_len` 压缩应当优先在 K 侧建立 index，而不是同时平均 K 和 V。K 的局部平滑性、小 block 结构和 address-like 几何更适合形成 block summary；V 的小 block 结构弱，说明直接把连续 V 平均成 compressed content 风险很大。更合理的第一版结构是：

```text
centered / whitened K block summary -> select candidate blocks
selected blocks -> gather original V or residual V content
```

第三，block 粒度应当偏小，并且 layer/head-aware。Qwen3 结果支持 4/8/16 token 的 K 小块结构，但不支持越大 block 越好。浅层 K 的 common direction 更强，后层 V 的内容维度更高，所以固定一个全层共享的 block size / rank / compression ratio 很可能不合理。

第四，raw K similarity 不能直接作为 index score。K 的 common direction 很强，raw dot product 或 cosine 可能主要反映公共方向，而不是可区分的检索信号。因此，真正的 K-index 实验应当比较：

```text
raw K
centered K
remove top-PC K
whitened K
```

第五，`seq_len` 压缩不能只有 semantic summary，还需要 exact residual path。数字、变量名、代码符号、引用位置这类信息可能不在 compressed summary 中稳定保留。当前几何结果也显示 V 侧 residual 不应被忽略。因此，理想结构更像：

```text
small K index -> sparse block selection -> exact / residual V read
```

而不是：

```text
large block average K/V -> replace original KV
```

## 4. 经典搜索工具如何映射到 KV Cache

经典搜索领域能提供的不是直接可搬的方案，而是一组问题分解方式。

### 4.1 Index 与 Content 分离

搜索系统通常不会拿完整文档直接做第一轮匹配，而是先构造索引：

```text
document -> index representation
query -> query representation
index score -> candidate set
candidate set -> rerank / read content
```

映射到 KV cache：

```text
historical tokens -> compressed/index KV entries
current hidden state -> indexer query
index score -> selected blocks
selected blocks -> real attention
```

CSA 的 Lightning Indexer 正是这个范式。

### 4.2 Candidate Recall 与 Exact Attention 分离

搜索系统关心第一阶段 recall：候选集必须覆盖真正有用的证据。KV cache 压缩也应该先问：

```text
在访问预算固定时，selector 是否能选到对输出有因果贡献的 KV？
```

这比“压缩向量重构误差低不低”更接近真实目标。

### 4.3 Multi-Stage Retrieval

搜索可以有多级：

```text
coarse block recall
-> fine token recall
-> exact attention / reranking
```

映射到长上下文模型，可以形成：

```text
document / segment level
-> block / page level
-> token / span level
-> exact KV attention
```

这样就可以把“百万 token context”理解成可逐级缩小的检索空间，而不是一次性塞进 flat attention。

### 4.4 Cache 与 Personalization

推荐系统和搜索系统都会缓存 hot queries、hot documents、user profile。LLM 系统也可以有类似结构：

- 当前会话的 hot KV blocks；
- 用户长期偏好的 memory summaries；
- 工具调用和网页内容的 local vector cache；
- 任务相关资料的 persistent memory index。

这里需要区分两类 cache：

```text
model-internal KV cache: 逐层逐头的激活记忆，服务于下一 token 预测
external knowledge cache: 网页、文档、工具结果的持久化语义索引
```

二者可以打通，但不能混为一谈。内部 KV cache 解决的是 decode-time memory bandwidth；外部 knowledge cache 解决的是系统级长期知识获取与复用。

## 5. 更大的想法：Context × Time × Harness = Ability

可以把大模型能力粗略理解成：

```text
ability = model parameters + context + tools + time + memory harness
```

参数提供基础能力，context 提供当前可见信息，tools 提供外部行动能力，time 提供多步搜索/推理预算，而 memory harness 决定系统能否把过去看到的信息持续组织起来。

从这个角度看，KV cache 压缩不只是节省显存。它的长期意义是：

> 让模型以可承受的成本拥有更宽广、更持久、更可检索的 context。

这也解释了为什么“把整个世界知识库都建 index”这个想法和 KV compression 有联系，但不是同一个层次的问题。

- 如果对象是全网网页、企业文档、代码仓库、用户历史，那是 external retrieval / RAG / agent memory；
- 如果对象是当前 prompt 在模型各层产生的 K/V 激活，那是 internal KV cache compression；
- 真正有想象力的系统可能会把两者连接起来：外部知识先被检索进 context，模型内部再把它压缩成可长期使用的 activation memory。

最终目标不是让模型每次都联网搜索，而是让系统越来越会沉淀：

```text
seen information
-> indexed external memory
-> retrieved into context when useful
-> compressed into internal working memory
-> updated into persistent cache after interaction
```

这会把“长上下文”从一次性窗口，变成一个不断增长、可维护、可检索的认知工作台。

## 6. 训练时方案与推理时方案的分叉

目前还不应该过早决定 sequence compression 一定发生在训练时，还是只发生在推理时。二者对应不同难度和收益。

### 6.1 训练时 compression

训练时把 compressor / indexer 放进模型结构，让模型从预训练或继续训练中学习：

```text
哪些 token 应该被压缩进 memory
哪些 compressed memory 应该被 query 访问
压缩后的 memory 如何保持 next-token prediction 能力
```

优点：

- 表征和压缩目标一致；
- 可以形成真正的 learned memory layout；
- 有机会从根本上减少模型对 flat KV 的依赖。

风险：

- 训练成本高；
- 需要从 causal consistency、batch training、decode equivalence 上重新验证；
- 如果任务没有足够长程检索压力，模型可能不会学到真正有用的 indexer。

CSA 属于这一类更强的方向。

### 6.2 推理时 compression

推理时在已有模型上做 KV 管理，例如：

```text
block summary
top-k block selection
KV quantization
cold block offload
token merging
attention sink / recent window retention
```

优点：

- 更容易落地；
- 不需要重新预训练；
- 可以直接和系统层 KV page / HBM-DRAM 管理结合。

风险：

- 模型没有被训练去适配这种 memory layout；
- selector 错误会直接造成不可恢复的信息丢失；
- 很难保证复杂任务中的 long-tail correctness。

### 6.3 混合路线

更现实的路线可能是：

```text
推理时诊断和原型
-> 找到 KV 可压缩/可索引的物理规律
-> 用继续训练或轻量训练让模型适配 indexed memory
-> 再把 compressor/indexer 固化为模型结构
```

也就是说，推理时方案可以作为 microscope，帮助我们识别模型内部 attention/KV 的结构；训练时方案才可能成为最终 architecture。

## 7. 下一步研究问题

当前最重要的不是马上设计一个 fancy compressor，而是回答几个约束问题。

### 7.1 KV cache 到底有什么可利用的物理特性？

需要从真实模型中测：

- attention mass 是否天然集中在少数 block；
- block-level summary 能否预测 token-level attention mass；
- 不同层/头的可压缩性是否不同；
- recent tokens、attention sink、实体位置、代码定义、工具返回内容是否有不同访问规律；
- K 的相似性结构和 V 的贡献结构是否一致。

### 7.2 什么样的 compressed entry 是有用的？

候选形式包括：

- fixed window weighted pooling；
- learned query-independent block memory；
- query-dependent dynamic compression；
- multi-vector block representation；
- key-only index + raw-value fallback；
- semantic slot memory；
- hierarchical block tree。

这里要避免只优化重构误差。真正的问题是：

```text
compressed memory 是否保留了未来 query 所需要的可检索证据？
```

### 7.3 Selector 应该学习什么？

CSA 的 indexer 是 query-to-block scoring。我们可以把 selector 目标拆成：

- 预测 attention mass；
- 预测 value contribution；
- 预测 CE delta；
- 预测某个 block 是否包含答案证据；
- 预测是否值得把 cold KV 搬回 HBM。

不同目标会导向不同结构。预测 attention mass 最接近原模型，预测 CE delta 更接近行为收益，预测搬运收益更接近系统目标。

### 7.4 如何保留精确信息？

任何 compression 都会遇到 long-tail precision 问题：

- 数字；
- 变量名；
- 代码符号；
- 引用位置；
- 表格单元；
- rare token；
- 用户明确给出的约束。

所以更合理的结构不是“所有信息都进 compressed KV”，而是：

```text
compressed semantic memory + retrievable exact residual memory
```

也就是高层 memory 负责找方向，底层 token/residual memory 负责取证据。

## 8. 一个暂定研究路线

可以把后续工作拆成四步。

### Step 1: 诊断现有 KV 的可索引性

先在已有模型上测 block-level selector 的上限：

```text
用真实 attention / loss contribution 作为 oracle
看 top-k block 能覆盖多少行为贡献
```

目标是回答：

```text
KV access 是否天然具有 block sparsity？
```

### Step 2: 训练轻量 indexer

在不改主模型的情况下，训练一个小 indexer：

```text
query hidden state + block summary -> block score
```

监督信号可以来自：

- attention mass；
- oracle ablation CE delta；
- evidence labels；
- future-token loss sensitivity。

目标是回答：

```text
cheap indexer 能否预测哪些 block 值得被 exact attention 访问？
```

### Step 3: 引入 compressed content memory

在 selector 有意义之后，再研究 compressed KV content：

```text
raw token KV
-> weighted block KV
-> multi-slot block KV
-> hierarchical memory entries
```

目标是回答：

```text
selected compressed KV 是否能替代 selected raw KV？
```

### Step 4: 训练时结构化 compression

如果前面证明 index/content 两层结构有效，再考虑更接近 CSA 的训练时方案：

```text
learned compressor
learned indexer
sparse compressed attention
residual exact-memory fallback
```

目标是让模型从训练开始就适配 indexed memory，而不是在推理时被动裁剪 KV。

## 9. 当前结论

这条线现在可以被重新命名为：

```text
From flat KV cache to indexed multi-resolution memory
```

它包含三个层次：

1. **模型内部层次**：sequence compression / hierarchical memory / CSA-style learned compressed KV；
2. **检索算法层次**：index-content separation / candidate recall / reranking / multi-stage retrieval；
3. **系统记忆层次**：HBM-DRAM-disk KV paging / external knowledge cache / persistent local memory。

我们之前的 U-Net Transformer 探索已经证明了一个方向：模型可能不需要在所有层都保持 full token-level memory。CSA 进一步说明，一个更直接的做法是显式训练 compressor 和 indexer，把 attention 拆成 cheaper retrieval 和 expensive content attention。

下一步不应直接追求“更 fancy 的压缩公式”，而应先建立 KV cache 的物理画像：

```text
什么信息被访问？
以什么粒度被访问？
哪些访问可以被 block index 预测？
哪些内容必须保留 exact residual？
哪些 compressed memory 对最终 loss/logits 真的有贡献？
```

只有回答这些问题，sequence compression 才能从架构灵感变成可验证的研究路线。
