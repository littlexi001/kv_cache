# KV Cache Graph Compression Project Overview

Date: 2026-06-04

## 0. One-Sentence Thesis

我们现在对 sequence compression / KV cache compression 的核心判断是：

> **LLM 的 KV cache 不应被理解成一串只能顺序扫描的 token buffer，而应被理解成一种可索引的 memory system：K 更像 address / index，V 更像 payload / evidence。只要 K-space 具有可建图的几何结构，长上下文 attention 就可以被拆成 `K-side retrieval` 和 `V-side faithful reading`。**

当前最有希望的方向是：

```text
K-indexed, V-faithful KV cache compression
```

也就是说，我们不是把 KV 一起粗暴平均，而是：

```text
1. 用 K 建 index / graph / cluster，快速找出相关历史区域；
2. 只在候选 token 上做 exact qK attention；
3. 对选中的 V 保持高保真读取。
```

## 1. 为什么这个问题值得做

长上下文推理的主要瓶颈之一，是每个新 query 都要和全部历史 K 做 full qK score。上下文越长，模型能看到的信息越多，但 attention 访问成本也越高。

直觉上，KV cache 其实是一种动态 memory。这个 memory 里现在放的是当前 sequence 的历史 token，但从系统角度看，它并不一定只能存当前 sequence：未来它也可能存网页、文档、工具结果、长期用户记忆、领域知识库，甚至是外部知识图谱压缩后的 memory entry。

因此，真正重要的问题不是“如何把一个 sequence 平均压短”，而是：

> **LLM 的 attention memory 能不能像搜索引擎或数据库一样被 index？**

如果可以，那么长上下文能力就不只是由 raw context length 决定，而是由更大的 memory pool 和更高效的 indexing 共同决定。

## 2. 我们已经观察到的 K/V 物理性质

我们先没有直接设计复杂模型，而是先问一个更基础的问题：

> **现有 LLM 的 K-cache 随 sequence 变长时，是否天然具有适合建图/检索的数学结构？**

到目前为止，实验支持四个稳定观察：

| 观察 | 含义 | 对方法的启发 |
|---|---|---|
| K 更像 index，V 更像 payload | K 更各向异性、更有局部平滑和 block/graph 结构；V 更 content-like | 应优先压缩/索引 K-side，V-side 尽量高保真读取 |
| K-space 有 residual graph geometry | 去掉 common direction 后，K 不是无结构散点，而是存在近邻、anchor 和区域结构 | K 可以建 graph / cluster / tree，而不只是顺序缓存 |
| K 图既有局部连续性，也有长程 shortcut | 高相似 K 不只是最近 token，还会跨越较长距离 | 单纯 sliding window 不够，需要 global index |
| layer/head 异质性很强，中后层更适合建图 | 浅层更局部、更不稳定；中后层 attention responsibility 更集中在少数 K-space region | index 应该 layer/head-aware，不能所有层用同一套策略 |

这四点把我们的方案从“拍脑袋稀疏化 attention”变成了一个有物理依据的问题：

```text
如果 K-space 已经像 address space，
那我们就应该用 address-space 的方法来组织它。
```

## 3. 我们验证了什么

在观察 K-space 几何之后，下一步问题是：

> **这种 K-space 图结构是否真的能帮助未来 query 找到 full attention 会使用的历史 token？**

我们做了两类验证。

第一类是 geometry / responsibility 验证：看 K-space 距离、近邻和 cluster 是否能预测 future query 的 attention responsibility。结果显示，中后层的 K-space cluster 对 future attention mass 有明显预测能力，深层尤其强；浅层则弱得多。

第二类是直接 sparse decode 验证：不只看 attention mass，而是把候选 token 真正用于 decode，比较 sparse attention 和 full attention 的 perplexity / loss。结果显示，`local + global cluster threshold` 可以在只访问约 20%-40% token 的情况下，达到接近 full attention 的效果。其中更经济的工作点大约访问 20% token，质量已经接近 full；更保守的工作点访问约 30%-40% token，几乎贴近 full。

这说明：

> **K-cache graph indexing 不是只在几何指标上好看，它已经在 decode-level 指标上表现出可用性。**

## 4. 当前 Solution 是什么

当前 prototype 可以概括为：

```text
prefill K -> cluster / graph index
decode query -> first query cluster centers
high-similarity clusters -> expand to all tokens inside cluster
local/sink tokens -> always keep
candidate tokens -> exact qK attention + faithful V read
```

更具体地说：

1. 对 prefill 阶段的 K-cache，在每个 layer / head 上建立 cluster。
2. 每个 cluster center 作为粗粒度 memory region 的 index entry。
3. decode 时，新 query 先和 cluster centers 比较相似度。
4. 只有相似度超过阈值的 cluster 会被展开。
5. 被选中的 cluster 内部 token 不再压缩，直接参与 exact qK attention。
6. local window / sink token 保留，避免丢掉局部语法、短程依赖和 attention sink。

这套设计里，每条策略都对应前面的物理性质：

| 策略 | 对应的物理性质 |
|---|---|
| 用 K 建 cluster | K 是 address-like memory，不是纯 content |
| 先查 cluster center | K-space 有 region / anchor / graph 结构 |
| 用 threshold 而不是固定 top-k | 每个 query 需要读取的 memory region 数量不应固定 |
| cluster 内全读 | cluster 只是 coarse index，真正 attention 仍应高保真 |
| 保留 local window | K 图有长程 shortcut，但模型仍需要局部连续性 |
| V 不做粗压缩 | V 是 payload，错误压缩会直接损害生成内容 |

所以这个方法不是“随便 cluster 一下试试”，而是从当前观察到的 K/V 结构自然推出的。

## 5. 当前 Promising 在哪里

这个方向最有价值的地方有三点。

第一，它已经从物理观察走到了 decode-level prototype。我们不只证明了 K-space 有图结构，也初步证明了这种图结构可以用于实际 sparse attention。

第二，它天然连接传统 search / database indexing。K-cache compression 不再只是神经网络里的矩阵近似问题，而是可以借鉴 inverted index、ANN、cluster routing、tree search、learned index、cache hierarchy 等系统思想。

第三，它给出了清晰的下一代模型结构方向。如果 inference-time graph index 已经有效但还不够好，那么更强的方案很可能是在训练时让模型学会产生更适合 indexing 的 K-space，而不是在推理时事后补救。

当前对外可以说的结论是：

> **我们已经找到了一条从 attention memory 的物理结构出发，到可运行 sparse attention prototype 的路径。现阶段主要问题已经从“这个方向有没有道理”变成“如何把 indexing 做得更准、更快、更适合训练”。**

## 6. 当前局限性

现在的方法还不是最终系统，主要局限可以分成三类。

### 6.1 Algorithmic Limitations

当前 cluster 是一个相对粗糙的 inference-time index。它没有在训练时参与 loss，也没有保证 cluster center 一定对应模型真正需要的 attention region。

因此它可能出现：

```text
cluster 过粗 -> 漏掉重要 token
cluster 过细 -> candidate 太多，压缩收益下降
统一 threshold -> 不适合所有 layer/head/query
pure cluster -> 丢掉局部上下文后质量崩坏
```

其中 `pure cluster` 已经被实验否定；当前有效形态必须是：

```text
local + global cluster threshold
```

### 6.2 Training-Inference Mismatch

当前模型训练时使用的是 full attention，并没有被要求形成一个特别适合 graph retrieval 的 K-space。我们在推理时强行建图，本质上是在利用 pretrained K-space 中自然出现的结构。

这会限制上限：

```text
模型没有被训练成“先查 index，再读 content”；
K-space geometry 只是自然涌现，不一定为 sparse retrieval 最优；
某些 head/layer 可能完全不适合 inference-time indexing。
```

所以更强的下一步可能是 train-time 方法：

```text
index-aware attention
cluster-aware KV cache
learned K-side router
DeepSeek CSA/MCA-style trainable compressor
graph-regularized K-space
```

### 6.3 Engineering / Systems Limitations

即使算法上能减少访问 token 数，工程上也不自动更快。

主要风险是：

```text
建 cluster 本身可能很贵；
query 和 cluster center 比较也有额外开销；
candidate mask / gather 在 CPU 侧可能很慢；
稀疏访问会破坏 GPU/MPS 对 dense matmul 的高效优化；
不连续 memory gather 可能比直接 full attention 更难优化。
```

也就是说，当前方法证明的是 algorithmic compression potential，不等于已经证明 wall-clock speedup。

系统优化要单独做，可能需要：

```text
GPU-side clustering / routing
static prefill index + incremental update
block-level contiguous layout
specialized sparse attention kernel
layer/head-selective sparse execution
```

## 7. 下一步资源应该投在哪里

下一步不应该继续无目的调参，而应该围绕三个问题推进。

### 7.1 把 Inference-Time Prototype 做成更强的 Upper Bound

目标是回答：

> 在不训练模型的情况下，K-side indexing 最多能做到什么质量/压缩比？

具体包括：

```text
layer/head-specific threshold
query-adaptive threshold
更好的 cluster / tree / anchor construction
block-contiguous candidate layout
longer、更难、真实长程依赖数据
```

### 7.2 判断是否必须进入 Training-Time Method

如果 inference-time 方法继续受限，最可能的解释不是方向错了，而是：

> 当前 pretrained model 没有被训练成 sparse-indexable memory。

这时下一步就应该设计训练时结构或 loss，让模型从一开始就学会：

```text
K side produces searchable addresses
cluster/root summarizes memory region
query first routes to region
V side keeps high-fidelity payload
```

### 7.3 判断系统瓶颈在哪里

算法压缩比和真实推理速度是两件事。

我们需要拆开测：

```text
cluster build time
cluster center query time
candidate gather/mask time
exact qK on candidates time
V read time
end-to-end decode latency
```

如果瓶颈主要在 CPU-side routing 和 gather，那么它是工程问题，可以通过 kernel/layout/system design 改。如果瓶颈来自必须访问太多 token 才能保持质量，那就是算法问题，需要更好的 index 或训练时约束。

## 8. 当前 Claim Boundary

现在可以说：

```text
K-cache has graph-friendly address geometry.
K 和 V 应该区别对待：K indexed, V faithful.
K-side cluster/graph routing 已经在 decode-level 指标上显示可行性。
local + global threshold routing 是当前最有效的 inference-time prototype。
```

现在还不能说：

```text
当前 naive implementation 已经比 full attention 更快。
当前 cluster 方法就是最终方案。
所有 layer/head 都适合相同 indexing 策略。
不需要训练时改模型。
```

当前最准确的项目状态是：

> **方向已经从 idea 进入 proof-of-principle：K-cache 确实可以被当作可建图的 memory address space。接下来真正有价值的工作，是把这个原理推进成更强的 trainable indexing architecture 或更高效的 sparse-attention system。**

## 9. Round Ledgers

详细实验、失败案例、脚本和数值结果保留在 round-level 文档中：

```text
fdong_seq_compress/k_cache_graph_round1_findings.md
fdong_seq_compress/k_cache_graph_round2_handoff.md
fdong_seq_compress/k_cache_graph_round3_todo.md
fdong_seq_compress/k_cache_graph_round4_threshold_cluster_findings.md
```

