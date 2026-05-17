# Inverse KV Indexing: Feature-Based Memory Layout

## 1. 核心问题

> 如果数据本身由 hierarchical, compositional features 生成，我们能否设计一种 MoE / KV 结构，使模型主动学习到 feature-based routing，并让同 feature token 形成可用于 KV reverse indexing 的 memory bucket？

## 2. 先验与假设

### 2.1 语言数据

> 人类语言数据是 hierarchical, compositional 的 sequential data。

- 低层 token 组合成短语、实体、事件、变量、函数、约束；
- 高层 feature 在长文本中反复出现，并通过指代、引用、因果、调用关系连接；
- 不同位置的 token 可能距离很远，但共享同一个 semantic / functional feature；
- 推理时真正需要访问的历史信息，往往不是“最近 token”，而是“和当前 query 共享相关 feature 的 token”。

因此，KV memory 的自然组织方式可能不是：

```text
token_1, token_2, ..., token_N
```

而更像：

```text
feature_bucket_1 -> tokens carrying feature 1
feature_bucket_2 -> tokens carrying feature 2
...
feature_bucket_M -> tokens carrying feature M
```

### 2.2 Attention 假设

标准 attention 可以被理解成一种 retrieval / search 过程：

```text
current query q
-> compare with historical keys
-> retrieve weighted values
```

如果语言数据确实由 hierarchical, compositional features 生成，那么 attention 不应只是按时间顺序平均读取历史 token，而应当对 feature 相关的历史 token 产生更高相关性。

也就是说，对 token $i,j$，如果它们共享某种 ground-truth feature：

```text
same slot
same higher-level unit
same entity / variable / function / semantic role
```

那么我们期望：

$$
q_i^\top k_j
$$

或 attention mass 相比不共享 feature 的 token pair 更高。

这一定义把 attention 从 dense sequence scan 重新解释成 feature-sensitive retrieval：

```text
query token
-> identify relevant feature
-> retrieve historical tokens carrying related feature
```

如果这个假设成立，那么 attention 内部已经包含某种可被利用的 retrieval structure。后续 MoE / gating 的目标不是凭空创造 feature index，而是把 attention 已经学到的 feature-sensitive retrieval structure 显式读出来，并转化成可用于 KV memory organization 的 routing label。

### 2.3 MoE 假设

在 MoE 模型中，gate/router 接收 token hidden state，并把 token 分发给 expert：

$$
g(h_i) \rightarrow e_i
$$

通常我们把这个过程理解成 compute routing：不同 token 被送到不同 FFN expert，以提升模型容量和计算效率。

但这里我们关心的不是已有 MoE gate 是否自然产生了这个现象，而是把 MoE gate 作为一种可设计的结构假设：

> MoE gate 不应只被设计成 compute router，也可以被设计成 feature router：它按照 token hidden state 中的 latent feature 对数据进行分组。

如果模型在合成的层次化数据上学会了这种 routing，那么 expert assignment 可以被看成一种 learned feature index：

$$
h_i \mapsto \text{feature bucket}
$$

这进一步导出一个可验证推论：

> 被分到同一个 expert 的 token，应该在 attention 计算中表现出更高相关性。

形式化地说，对 token $i,j$，如果：

$$
e_i = e_j
$$

那么我们期望：

$$
q_i^\top k_j
$$

或 attention mass 相比不同 expert 的 token pair 更高。

### 2.4. Inverse KV 的直觉

标准 attention 可以看成：

```text
current query q -> scan all historical K/V -> weighted read V
```

也就是每个 query 都在平铺 KV cache 上做一次全量检索。

`inverse_kv` 想反过来问：

> 如果模型被设计成必须识别当前 token 属于哪些 feature，那么能否用 gate / expert assignment 作为 KV cache 的索引入口？

也就是说，KV cache 不再只是按时间顺序存储：

```text
K/V[time]
```

而是额外组织成：

```text
K/V[expert][time]
```

推理时，query 可以先通过 gate 得到相关 expert，再优先检索同 expert 或相关 expert 的 historical K/V。

如果成立，这会得到一种 feature-indexed KV memory：

```text
query hidden state
-> gate predicts relevant feature buckets
-> retrieve K/V from selected buckets
-> exact attention inside selected candidates
```

这里 KV cache 节省不是最初目的，而是 byproduct：模型不再需要对所有历史 token 做 flat scan，而是先通过 feature index 缩小候选集合。


## 3. 第一轮实验：Attention/MoE 是否形成 feature bucket

这一轮实验的目的不是验证新的 gating 机制，而是先测试一个基准问题：

> 在具有明确 hierarchical compositional structure 的合成数据上，标准 Attention + MoE 是否会自然学出我们希望的 feature-based routing？

### 3.1 核心结论

**结论 1：模型能建模认为构造的 slot sequential 数据。**

训练后 next-token loss 明显下降，token accuracy 约为 90% 以上。这说明模型确实学到了合成数据中的可预测结构。

**结论 2：Attention 会捕捉 sequential structure，但正确 slot 上的权重只有 20%。**

同 slot token pair 的 attention score 明显高于 non-slot token pair，说明 attention 确实对 ground-truth slot structure 敏感。

但从 attention mass 看，一个 token 分给历史同 slot token 的总注意力只约为 20%。这说明 attention 使用了 slot 信息，但并没有以“主要注意力都集中到同 slot token”这种强形式组织计算。

**结论 3：标准 MoE router 几乎没有自然形成 feature slot / index。**

MoE expert assignment 与 ground-truth slot id 的对齐很弱。同 slot token 不会稳定进入同一个 expert；同 expert token 的 attention 相似性也不强。

因此，标准 MoE router 不能直接被解释为可用于 KV retrieval 的 feature index。

### 3.2 数据生成方法

数据直接由 token id 序列生成，不依赖自然语言文本。

最小结构单元是固定长度 slot，例如 `block_size = 4` 时，一个 slot 可以是：

```text
1 2 3 4
```

数据集初始化时会先固定一个 slot pool。之后所有训练 step、所有 batch 都从同一个固定 grammar / slot pool 中采样，因此不会出现同一个 seed 下 slot pattern 一会儿是 `1 2 3 4`、一会儿变成 `4 3 2 1` 的情况。

更高层 hierarchy 通过组合低层 unit 构造。例如：

```text
layer 0: token -> slot
layer 1: slot -> higher-level unit
```

每条训练样本是一个按 hierarchy grammar 展开的 token id 序列。训练目标仍是标准 causal language modeling：

```text
input  = S[0:n-1]
target = S[1:n]
```

这一设计保证数据本身具有明确的 ground-truth feature / slot structure，可以离线分析：

- 每个 token 属于哪个最小 slot；
- 每个 token 属于哪个 higher-level unit；
- attention 是否偏向同 slot token；
- MoE expert assignment 是否和 slot id 对齐。

### 3.3 训练模型

当前第一轮实验使用的是标准小型 causal Transformer，而不是新的 routing 架构。

主要设置：

- 3 层 Transformer；
- 标准 causal self-attention；
- FFN 替换为标准 top-1 MoE；
- 4 个 unique experts；
- 不使用 common expert；
- hidden size 为 128；
- synthetic sequence length 为 128；
- 最小 slot 长度为 4；
- 使用普通 next-token prediction loss 训练。

因此，这一轮实验检验的是：

> 标准 MoE router 在没有额外约束时，是否会自然变成 feature / slot router？

### 3.4 对当前假设的修正

第一轮实验削弱了原始强假设：

```text
语言数据具有 hierarchical features
=> 标准 attention 和标准 MoE 会自然暴露这些 features
=> MoE expert id 可以直接作为 KV cache index
```

更合理的修正版本是：

```text
语言数据具有 hierarchical features
=> 模型可能会利用这些 features
=> 但标准 attention / 标准 MoE 不一定以可索引、可路由的形式显式暴露它们
```

尤其需要注意标准 Transformer block 的计算顺序：

```text
hidden state
-> attention
-> MoE / FFN
```

标准 MoE router 发生在 attention 之后，因此即使它携带 feature 信息，也不能直接指导同一层 attention 的 KV indexing。它最多能影响后续层，不能作为当前层 attention 前的 candidate selector。

因此，下一步不应继续假设“复用现有 MoE router 即可”，而应测试新的 gating / routing 机制：

> routing 必须发生在 attention retrieval 之前，并且需要被显式训练成 retrieval-aligned feature router。

## 4. 第二轮实验：验证 Attention 的有效 token 集中性与新的 MoE gating 机制的 selectivity

第一轮实验说明，标准 MoE router 不会自然形成可用的 feature index。因此第二轮实验不再假设现有 MoE 可以直接复用，而是拆成两个更具体的问题：

```text
1. 当前 attention 是否正确捕捉到 slot-structured sequential feature？
2. 能否设计新的 MoE gating，使 routing 与 attention 中的 feature pattern 对齐？
```

这轮实验的目的不是马上证明 KV cache 可以被压缩，而是先验证 attention 和 routing 是否能在同一个 feature structure 上对齐。

### 4.1 假设 A：当前 attention 是否已经足够支持 slot-structured feature retrieval

第一轮实验中，同 slot token 的 attention score 高于 non-slot token，但同 slot token 只占总 attention mass 的约 20%。这带来一个关键问题：

> 虽然 same-slot attention mass 不高，但这些 attention 是否已经包含模型预测所需的核心信息？

如果只保留 same-slot KV 后模型仍然能接近 full attention 的预测效果，则说明当前 attention 已经形成了可用的 feature retrieval，只是 attention mass 看起来不集中。

如果只保留 same-slot KV 后效果显著下降，则说明模型还依赖其他结构，例如 higher-level unit、slot boundary、position continuation 或普通 dense context。此时需要先设计新的 attention 范式，让 attention 更明确地捕捉 hierarchical / compositional feature。

实验设计：

- full attention：保留完整 KV，作为上界；
- only same slot：只保留与 query 属于同一最小 slot 的 historical KV；
- only same higher-level unit：只保留与 query 属于同一高层 unit 的 historical KV；
- same slot + same higher-level unit：同时保留两类 feature-related KV；
- random same-size KV：保留相同数量的随机 KV，作为 size-matched baseline。

评估指标：

- next-token loss / token accuracy；
- retained attention mass；
- sparse attention output 与 full attention output 的差异；
- same-slot sparse KV 是否显著优于 random same-size KV；
- same-slot + higher-level sparse KV 是否接近 full attention。

这一组实验用于判断：

> attention 中的 feature structure 是真正可用于预测的 retrieval structure，还是只是一个弱相关统计现象。

实验结果：

| attention mask | loss | accuracy | visible KV |
|---|---:|---:|---:|
| full attention | 0.261 | 91.40% | 100% |
| same local slot occurrence | 2.022 | 74.43% | 9.46% |
| same local slot pattern | 1.978 | 75.08% | 11.25% |
| same higher-level unit | 0.264 | 91.17% | 26.36% |
| random same-size KV | 3.808 | 52.93% | 11.25% |

核心结论：

```text
lowest-level slot 不是当前模型的充分 retrieval bucket；
higher-level unit 几乎可以替代 full attention。
```

按位置拆开看，same local slot 对 slot 内部预测仍然有效，但对跨 slot boundary 的预测严重不足：

| mask | internal accuracy | boundary accuracy |
|---|---:|---:|
| full attention | 99.58% | 67.85% |
| same local slot | 91.86% | 23.40% |
| same higher-level unit | 99.38% | 67.55% |

这说明当前模型真正依赖的 feature 粒度不是最低层 slot，而是 higher-level unit。更准确地说：

> 现有 attention 已经形成了可用的 higher-level feature retrieval structure。

进一步分析 full attention 下的 attention mass：

| layer | local slot history mass | higher-level history mass | higher-level baseline |
|---|---:|---:|---:|
| layer 0 | 18.47% | 33.94% | 24.45% |
| layer 1 | 17.53% | 47.63% | 24.45% |
| layer 2 | 17.74% | 47.07% | 24.45% |

因此，higher-level unit 并没有吃掉接近 100% 的 attention mass；但它只占约 26% 的可见 KV，却承载了约 47% 的历史 attention mass，并且只保留这部分 KV 时推理性能几乎不下降。

分 head 看，没有出现强烈的 head specialization。不同 head 都更偏向 higher-level unit，而不是某些 head 专看 local slot、某些 head 专看 higher-level unit。layer 1 和 layer 2 对 higher-level unit 的偏向明显强于 layer 0。

### 4.1.1 标准 MoE 对 higher-level unit 的集中性

由于 same higher-level unit 几乎可以替代 full attention，接下来检查标准 MoE router 是否也在 higher-level unit 上形成集中分发。

结果如下：

| layer | same higher-level unit -> same expert | slot-expert MI | slot -> expert purity |
|---|---:|---:|---:|
| layer 0 | 29.57% | 0.117 | 42.93% |
| layer 1 | 27.87% | 0.102 | 40.24% |
| layer 2 | 27.19% | 0.091 | 40.03% |

由于当前模型只有 4 个 expert，按 expert load 随机分配时，同 expert 概率本身就在约 25% 到 27% 附近。因此上述结果只比随机负载基线略高，不能说明标准 MoE router 明确按照 higher-level unit 聚类。

同时，same expert token 的 attention 相似性仍然很弱：

| layer | higher-level attention mass | same expert attention lift |
|---|---:|---:|
| layer 0 | 33.67% | 1.11x |
| layer 1 | 47.26% | 1.06x |
| layer 2 | 46.70% | 1.08x |

因此，目前最重要的结论是：

```text
attention 已经学到了 higher-level feature retrieval；
标准 MoE router 没有把这个 retrieval structure 显式读出来。
```

这正是第二轮后续 MoE 设计实验的动机：新的 gating 不应继续复用标准 MoE 的 router 输入和 router 目标，而应尝试从 attention output 或 head-level attention representation 中读出已经存在的 feature structure。

### 4.2 假设 B：使用纯 attention output 做 routing 是否更容易对齐 attention pattern

标准 Transformer block 中，MoE router 看到的输入通常包含 residual：

```text
router input = x + attention_output
```

这可能让 router 主要依据 token identity、position 或 residual shortcut 做分发，而不是依据 attention 真正读出的 feature 信息。

因此提出新的 gating 机制：

```text
attention_output = Attention(norm(x))
router_logits = Router(attention_output)
expert_input = x + attention_output
expert_output = MoE(expert_input, route = router_logits)
```

也就是说：

- 用不加 residual 的 attention output 决定 routing；
- 送入 expert 的表征仍然使用加 residual 后的表征；
- 这样只改变 routing signal，不破坏 Transformer block 的主干信息流。

这一实验验证：

> 如果 router 只看 attention 读出的信息，它是否更容易和 attention 的 same-slot / same-feature pattern 对齐？

评估指标：

- expert assignment 与 slot id / higher-level unit id 的 mutual information；
- same-slot token 是否更稳定进入同一 expert；
- same-expert token 的 attention score / attention mass 是否高于 different-expert token；
- 与标准 MoE router 的结果对比；
- 与 shuffled expert assignment baseline 对比。

### 4.3 假设 C：head-level MoE 是否更适合捕捉 feature-specific routing

不同 attention head 可能学习不同结构，例如：

- local slot pattern；
- slot boundary transition；
- higher-level unit；
- position continuation；
- copy / induction pattern。

如果先把所有 head concat 后再做 token-level routing，router 看到的是混合后的表征，可能难以和任意单个 head 的 attention pattern 对齐。

因此提出 head-level MoE：

```text
head_output[h] -> Router_h -> Expert_h
```

每个 attention head 拥有自己的 router 和一组小 expert。routing 表征只来自当前 head 的 output，而不是所有 head concat 后的 token representation。

这一实验验证：

> MoE routing 是否需要按 attention head 分解，才能和 head-specific feature geometry 对齐？

评估指标：

- 每个 head 的 same-slot / same-feature attention mass；
- 每个 head 的 expert assignment 与 slot / higher-level unit 的对齐程度；
- 同一 head 内 same-expert token 的 attention score 是否更高；
- 哪些 head 更偏向 low-level slot，哪些 head 更偏向 high-level unit。

### 4.4 假设 D：纯 attention output routing 与 head-level MoE 可以组合

如果 4.2 和 4.3 分别有效，则可以组合成更强结构：

```text
per-head attention_output[h]
-> Router_h(attention_output[h])
-> Expert_h(residual-added head representation)
```

这个结构同时满足：

- routing signal 来自纯 attention output；
- routing 粒度与 attention head 对齐；
- expert 输入仍然保留 residual 信息。

这一实验验证：

> 当 router 输入和 routing 粒度都与 attention 更一致时，MoE 是否能形成更强的 feature-indexed routing？

实验顺序应当保持可解释性：

```text
standard MoE
-> pure-attention-output router
-> head-level MoE
-> pure-attention-output + head-level MoE
```

只有这样才能判断性能变化来自哪一个设计因素。

### 4.5 第二轮实验的判定标准

第二轮实验成功不要求马上实现 KV cache 压缩，而要求看到以下链条成立：

```text
attention learns feature-sensitive retrieval
and
new router can read out / align with this feature-sensitive retrieval
and
expert assignment predicts attention-relevant KV candidates better than random
```

如果成立，下一步才进入真正的 architecture 阶段：

```text
把 retrieval-aligned router 前置到 attention 之前，
用它做 KV candidate selection，
再验证 sparse KV inference 是否接近 full attention。
```

如果不成立，则说明问题不只是 MoE router，而是 attention 本身没有以足够可索引的方式组织 feature memory，需要先设计新的 attention 训练目标或 attention mask / retrieval objective。

## 5. 当前阶段性结论

当前工作的目标可以概括为：

> Attention 和 MoE 都正确捕捉 hierarchical / compositional / Zipf-distributed feature；MoE routing 进一步成为 attention KV cache 的 reverse index。

迄今实验说明：这个目标中的若干子假设并不等价。模型确实会利用 hierarchical feature，但这些 feature 在不同模块中的呈现形式不同。

### 5.1 Attention score 如何捕捉 feature

Attention 明确捕捉到了 hierarchy，尤其是 higher-level unit。

在合成 hierarchy 数据上，只保留 same higher-level unit 的 KV 时，推理 loss / accuracy 几乎不下降；只保留 same local slot 则明显掉点。这说明模型真正依赖的 retrieval bucket 不是最低层 slot，而是 higher-level compositional feature。

但 attention 捕捉到的 feature 不够干净：

- higher-level unit 只占部分 attention mass，但几乎足以保持预测性能；
- 不同 hierarchy 信息混在不同 layer / head 中；
- 没有出现某些 head 专门看 local slot、某些 head 专门看 higher-level unit 的清晰 specialization；
- high-level feature 主要体现在 token-to-token attention relation，而不是单 token hidden state cluster。

因此，当前最可靠的 feature carrier 是：

```text
attention score / attention relation
```

而不是 hidden state 中一个直接可分的 high-level feature vector。

### 5.2 MoE gating 如何捕捉 feature

标准 MoE router 不能自然形成可用的 feature index。

实验中，MoE expert assignment 与 ground-truth feature 的对应较弱：

```text
local slot 对齐：约 35% ~ 43%
higher-level unit 对齐：约 28% ~ 34%
```

这说明 MoE 有一定 selectivity，但不能解释为“同一 feature 稳定进入同一 expert”。

新的结构带来了更明确的信号：

```text
attention output w/o residual routing + head-level MoE
```

可以显著提高 MoE routing 与 attention 高相关 token 的一致性：

```text
uniform 数据：include-self same-expert attention mass 约 62.5%
Zipf 数据：include-self same-expert attention mass 约 59.5%
history-only same-expert attention mass 约 40%
```

因此当前结论是：

> MoE 不能独立发现干净的 ground-truth hierarchy；但当 routing input 更接近 attention output，且 routing 粒度下沉到 head-level 时，MoE 可以部分读出 attention 已经学到的 relational feature。

也就是说，MoE routing 更像是 attention feature 的 readout，而不是 feature 的 first discoverer。

### 5.3 参数矩阵、表征空间与 Zipf

Zipf 分布确实影响模型，但影响方式不是“高频 high-level feature 自动占据参数矩阵最大奇异方向”。

首先，raw SVD 的最大奇异方向大量包含 common / mean direction：

- embedding raw top singular direction 几乎等于 embedding mean direction；
- representation raw top singular direction 经常接近 representation mean direction；
- final representation 的 top direction 与 embedding mean 稳定相似。

因此，参数矩阵或表征矩阵的最大奇异方向不能直接解释为 semantic / hierarchy feature。

去掉 mean direction 后，出现了更细的结构：

- local slot 在 hidden representation 中很可分；
- higher-level unit 在 hidden representation 中不干净；
- Zipf frequency 对 local-slot feature 更明显；
- 高频 local slot 更容易对齐 `k_proj`、MoE `gate_proj / up_proj` 的重要方向；
- high-level compositional feature 仍然没有稳定占据参数矩阵 top singular directions。

因此更准确的 feature 图景是：

```text
local / frequency feature
-> 更像 vector-space feature
-> 可以体现在 representation norm、k_proj、MoE gate/up 参数方向中

higher-level / compositional feature
-> 更像 relational feature
-> 主要体现在 attention score / attention pattern 中
```

这说明 Zipf distribution 会塑造表征和参数方向，但它首先影响的是低层可复用 feature，而不是自动生成干净的 high-level feature basis。

### 5.4 MoE 能否作为 inverse KV index

以当前标准 Transformer block 来看，MoE 不能直接 serve as inverse indexing of attention KV cache。

原因有两个：

1. 标准 block 中 MoE routing 发生在 attention 之后，因此同一层 MoE gating 不能反过来指导已经发生的 attention KV selection。
2. MoE routing 与 ground-truth hierarchy 的对应不够干净，不能直接作为 reliable KV bucket id。

但当前实验并没有否定 inverse KV 的方向，而是修正了路线：

```text
不能指望标准 MoE router 自然变成 KV index；
更合理的是先让 attention 学出干净 feature relation，
再把 attention correlation 显式转化为 routing / indexing signal。
```

### 5.5 下一步 TODO

1. 设计新的 attention 约束，让不同 layer / head 捕捉更干净、更可分离的 hierarchy feature。

2. 把 attention score / attention-derived correlation 显式作为 routing index，而不是期待 MoE 从 hidden state 中自己发现 hierarchy。

3. 在新的 routing index 前置到 attention 之前后，再验证它是否能做 KV candidate selection，并在 sparse KV inference 下接近 full attention。
