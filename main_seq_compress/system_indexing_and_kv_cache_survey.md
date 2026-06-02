# 数据库 / 搜索系统中的 Indexing 思想与 LLM KV Cache 管理调研

Date: 2026-06-02

## 0. 结论先行

数据库、搜索、向量检索系统对 indexing 的核心理解是：

> index 不是压缩数据本身，而是为了避免 full scan 而额外维护的可查询组织结构。它把一次昂贵的全量访问，变成 cheap candidate recall + expensive exact read / rerank。

映射到 LLM KV cache：

```text
传统系统:
  index -> candidate rows / documents / vectors -> exact read / rerank

KV cache:
  K-side index -> candidate tokens / blocks -> exact qK over candidates -> high-fidelity V read
```

现有 LLM systems 对 KV cache 的理解主要有五类：

| 系统视角 | 代表工作 | KV cache 被看成什么 | 和我们课题的关系 |
|---|---|---|---|
| memory paging | vLLM PagedAttention, vAttention | 动态增长的 GPU memory pages | 解决存储碎片和分配，不解决“哪些 token 值得 attend” |
| prefix / chunk reuse | SGLang RadixAttention, CacheBlend, LMCache | 可跨请求复用的 prefix/chunk state | 解决重复 prefill，不解决单条长序列内部 sparse retrieval |
| tiered storage / offloading | LMCache, CacheGen, InfiniGen | 可在 GPU/CPU/SSD/network 间移动的数据对象 | 解决数据搬运和容量，不直接建立 attention-relevant index |
| eviction / compression | H2O, SnapKV, CacheGen 等 | 可丢弃或压缩的历史 token state | 解决保留哪些 token，但常依赖 attention heuristic |
| retrieval index | RetrievalAttention, vector ANN 类工作 | 可通过 ANN/vector search 检索的 K/V vectors | 和我们最接近，但还需要处理 layer/head、RoPE、query-attention recall |

对我们当前课题最重要的启发是：

> 不要把 K-cache graph index 直接理解成“做一个 HNSW”。更稳的路径是借鉴系统领域的两阶段原则：先用 K-side lightweight index 做 candidate recall，再用真实 qK/V 做 exact verification。

当前最值得做的下一步仍然是：

```text
K-K graph candidates
vs
full qK attention
measure attention mass recall / top-token recall / CE delta
```

只有这个 gate 通过，才值得进一步设计 inference-time graph index 或 train-time learned indexer。

## 1. 传统系统如何理解 Index

### 1.1 B/B+ Tree：把随机查找变成层次化导航

B-tree / B+tree 是数据库最经典的 ordered index。Comer 的综述把 index 类比成文件柜上的标签：index 指向文件中更小的区域，使查询不必逐项扫描。B-tree 的关键是：

```text
sorted key hierarchy
high fan-out
page-aware layout
logarithmic lookup
range scan support
```

来源：[The Ubiquitous B-Tree, Comer 1979](https://g-trees.github.io/g_trees/assets/references/comer1979ubiquitous.pdf)。

对 KV cache 的启发：

```text
K graph / K tree 不应只看数学距离。
它还必须考虑 page / block / GPU memory layout。
```

也就是说，真正系统化的 K index 可能需要两层：

```text
logical K index:
  which token/block should be considered?

physical KV layout:
  where is this token/block stored and how cheaply can it be gathered?
```

这正好对应我们现在的分工：

```text
K-side address index
V-side faithful payload read
```

### 1.2 LSM Tree：在线插入、后台合并、冷热分层

LSM-tree 解决的是 high insert-rate 场景。它不在每次写入时都随机更新大树，而是：

```text
new entries first go to memory component
flush to sorted immutable runs
background merge / compaction
use batching to trade write cost for read amplification
```

原始 LSM paper 强调的核心收益来自 batch merge 和 multi-page I/O，而不是单次随机更新。来源：[The Log-Structured Merge-Tree](https://db.cs.berkeley.edu/cs286/papers/lsm-acta1996.pdf)。

对 KV cache 的启发：

LLM decode 的 KV cache 也是 streaming insert：

```text
t -> t+1 -> t+2
```

每步插入一个新 token 的 K/V。一个在线 K index 不能假设可以离线重建完整图。更实际的结构可能是：

```text
recent hot buffer:
  exact local window, cheap append

warm graph/index:
  periodically updated K graph or anchors

cold storage:
  compressed/offloaded old KV blocks
```

这类似 LSM 的 memory component + disk component，只是 KV cache 的读是 attention retrieval，而不是 key lookup。

### 1.3 Inverted Index：从 content term 到 candidate documents

搜索引擎的 inverted index 把：

```text
term -> postings list(doc_id, position, weight, ...)
```

这样 query 不需要扫描所有文档，只需要合并 query terms 对应的 postings。Google 早期搜索系统就是通过 crawl、forward index、sorter、inverted index、ranking 等阶段组织 web-scale retrieval。来源：[The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/)。

搜索系统还有一个重要原则：

```text
candidate recall first
exact scoring / ranking later
```

动态剪枝方法如 MaxScore / Block-Max WAND 会利用上界跳过不可能进入 top-k 的 postings block。Lucene 后来引入 Block-Max WAND 来提高 top-k retrieval 效率。来源：[From MaxScore to Block-Max WAND](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148045/)。

对 KV cache 的启发：

KV attention 也可以看成：

```text
query q_t
-> K-side candidate generation
-> exact qK score only on candidates
-> V-side weighted read
```

我们的 K graph index 如果要成为系统方法，也应该有类似 WAND 的思想：

```text
cheap bound / cheap route:
  decide blocks that cannot matter

exact qK:
  only compute on surviving candidates
```

### 1.4 Vector ANN：高维空间里用近似索引换 recall-latency tradeoff

传统数据库的 B-tree 对高维向量不适合。Meta Faiss 的介绍明确指出，向量检索需要处理：

```text
nearest neighbor under Euclidean distance
maximum inner product search
billion-scale vectors
GPU/CPU optimized k-selection
compressed indexes
```

来源：[Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)。

HNSW 是典型 graph-based ANN。它维护一个多层 small-world graph，查询时从高层粗导航到底层精搜索。来源：[HNSW paper summary](https://huggingface.co/papers/1603.09320)。

DiskANN 进一步说明，向量索引不是只关心算法复杂度，还关心：

```text
DRAM vs SSD
visited vertices cache
graph layout
recall / latency / memory density
```

来源：[DiskANN NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)。

对 KV cache 的启发：

我们的 K-cache graph 和 ANN 很像，但有几个关键差异：

| Vector DB ANN | LLM K-cache graph |
|---|---|
| 数据库通常较静态，可离线建索引 | KV cache 每个 token 在线增长 |
| query 是外部 embedding | query 是当前 layer/head 的 q_t |
| 目标是近邻 recall | 目标是 qK attention mass / CE delta |
| 向量没有 causal constraint | K-cache 必须 causal，只能访问历史 |
| 向量索引通常单空间 | K-cache 是 layer/head-specific 多空间 |
| 检索结果通常是文档 | 检索结果要接入 V-side attention aggregation |

所以 ANN 的正确用法不是照搬 HNSW，而是借它的系统思想：

```text
multi-scale graph
entry points / anchors
bounded candidate expansion
approximate recall then exact rerank
memory-layout-aware graph storage
```

## 2. 现有 LLM Systems 如何组织 KV Cache

### 2.1 vLLM PagedAttention：把 KV cache 当成 virtual memory pages

vLLM / PagedAttention 的问题定义是：KV cache 很大、动态增长、容易碎片化，并且 batch size 受 GPU memory 限制。它把 KV cache 切成 blocks，用类似 OS paging 的方式管理逻辑位置和物理 GPU block。论文摘要明确说 PagedAttention 受到 virtual memory / paging 启发，目标是减少 fragmentation 和 redundant duplication。来源：[PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180)。

理解方式：

```text
KV cache = dynamically allocated memory pages
index = block table / page table
goal = memory efficiency and sharing
not goal = semantic or attention-relevant token selection
```

对我们的启发：

```text
如果我们做 K graph index，最终必须落到 paged KV layout 上。
graph candidate 输出不能只是 token ids，还要能高效映射到 KV blocks。
```

### 2.2 vAttention：保留 contiguous virtual memory，依赖 demand paging

vAttention 认为不一定要改 attention kernel 来支持 paged blocks；它希望保留 contiguous virtual address，并通过系统 demand paging 做物理分配。来源：[vAttention paper](https://arxiv.org/abs/2405.04437)。

理解方式：

```text
KV cache = virtual contiguous region + physical on-demand allocation
goal = dynamic memory management without custom PagedAttention kernels
```

对我们的启发：

K index 的系统落地可能有两种路线：

```text
Route A:
  graph candidates operate on paged physical blocks

Route B:
  graph candidates expose logical positions and let virtual memory / kernel handle layout
```

### 2.3 SGLang RadixAttention：用 radix tree 做 exact prefix reuse

SGLang 的 RadixAttention 把多请求之间的相同 token prefix 组织成 radix tree，从而复用相同 prefix 的 KV cache。官方文档写得很直接：RadixCache 用 prefix tree / trie 保存 token sequence 到 KV cache indices 的映射。来源：[SGLang RadixAttention docs](https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention)。

理解方式：

```text
KV cache = reusable prefix states
index = radix tree over exact token sequences
lookup = longest prefix match
goal = avoid repeated prefill
```

对我们的启发：

RadixAttention 是 exact token-sequence index，不是 semantic K index。它说明了一个重要系统原则：

```text
KV cache index 可以不是向量 index。
它可以是 token-prefix index、block index、graph index、或者多者组合。
```

我们的问题更像：

```text
within one long sequence:
  which old K/V should current q_t attend?
```

RadixAttention 的问题更像：

```text
across requests:
  which prefix KV has already been computed?
```

### 2.4 LMCache：把 KV cache 变成跨 GPU/CPU/storage/network 的可编排对象

LMCache 认为 KV cache 不应只存在于 GPU memory。它把 KV cache 从 inference engine 中抽取出来，使其能跨 queries 和 engines 共享，并支持 GPU、CPU、storage、network 层的 orchestration。来源：[LMCache paper](https://arxiv.org/abs/2510.09665)。

理解方式：

```text
KV cache = reusable data object beyond a single request lifecycle
index = cache key / connector / placement metadata
goal = offload, transfer, reuse, orchestration
```

对我们的启发：

未来 K graph index 不应只问：

```text
which token is relevant?
```

还要问：

```text
where is its KV stored?
GPU, CPU, SSD, remote storage?
fetch cost how high?
```

这会自然引入 cost-aware candidate selection：

```text
score = attention_relevance - data_movement_cost
```

### 2.5 CacheBlend：RAG 场景下复用非 prefix KV chunks

CacheBlend 关注 RAG 中大量 retrieved chunks 重复出现但位置不一定相同的问题。它试图融合 cached KV knowledge，减少 full recompute，并报告 TTFT 和 throughput 提升。来源：[CacheBlend paper](https://arxiv.org/abs/2405.16444)。

理解方式：

```text
KV cache = reusable knowledge chunks
index = chunk identity / reuse metadata
problem = reused content may appear at different positions
goal = reuse chunk KV without full recompute
```

对我们的启发：

这提醒我们：KV cache indexing 有两种不同对象：

```text
content identity index:
  same text chunk reused across requests

attention geometry index:
  current q_t selects relevant historical K/V inside one context
```

我们的 K graph 更接近第二种。

### 2.6 CacheGen：把 KV cache 压成可传输 bitstream

CacheGen 把 KV cache 看成需要在系统中传输的 tensor 数据。它利用 KV cache distributional properties 设计 tensor encoder，并根据带宽调整不同部分的压缩等级。来源：[CacheGen paper](https://arxiv.org/abs/2310.07240)。

理解方式：

```text
KV cache = bandwidth-sensitive tensor stream
index = not primary object
goal = reduce context-loading delay and transfer size
```

对我们的启发：

如果 K graph index 最后需要跨 CPU/GPU/SSD 取旧 KV，传输压缩会成为第二阶段问题：

```text
first:
  decide which KV blocks to fetch

then:
  decide how to compress / stream fetched KV blocks
```

### 2.7 KVServe：在 disaggregated serving 中把 KV 当成 service-aware payload

KVServe 关注 PD separation / KV disaggregation 之后的通信瓶颈。它的关键观察是：一旦 KV state 跨 network / storage boundary 流动，KV 就成为端到端瓶颈；静态压缩配置无法适应 workload mix、bandwidth、SLO 和 quality budget 的变化。来源：[KVServe paper](https://arxiv.org/abs/2605.13734)。

理解方式：

```text
KV cache = communication payload under service constraints
index = not the main object
policy = online controller selects compression profile
goal = optimize latency / quality / bandwidth tradeoff
```

对我们的启发：

如果 K graph index 未来进入真实系统，candidate selection 也不应只看 attention relevance：

```text
candidate utility = attention relevance - movement/compression cost
```

这与数据库系统的 cost-based optimizer 很像。一个 token/block 是否值得取，不只取决于它是否相关，还取决于它在 GPU、CPU、SSD、remote storage 中的读取成本。

### 2.8 RetrievalAttention：最接近“KV vectors 建 ANN index”的路线

RetrievalAttention 明确提出为 KV vectors 建 ANN index，在 CPU memory 中检索 generation 时最相关的 KV vectors，从而加速 long-context attention。来源：[RetrievalAttention paper](https://arxiv.org/abs/2409.10516)。

理解方式：

```text
KV cache = vector database
index = ANN index over KV vectors
query = current attention query
goal = retrieve relevant KV tokens on demand
```

这是和我们最接近的系统方向。但它也暴露了我们必须小心的问题：

```text
ANN nearest neighbor recall != attention mass recall automatically
RoPE / layer / head / causal constraints matter
retrieval overhead must be lower than saved qK compute
```

我们的差异化问题可以更精确地写成：

> 在 Qwen3 K-cache 的 residual graph geometry 下，是否存在比 generic ANN 更 layer/head-aware、更 attention-recall-aware 的 K-side graph index？

### 2.9 H2O / InfiniGen：从 cache policy 角度管理 token importance

H2O 把 KV cache 管理成 eviction policy：保留 recent tokens 和 attention heavy hitters。来源：[H2O paper](https://arxiv.org/abs/2306.14048)。

InfiniGen 关注 dynamic KV cache management，并和 offloading 系统协同。来源：[InfiniGen paper](https://arxiv.org/abs/2406.19707)。

理解方式：

```text
KV cache = limited-capacity cache
policy = retain recent + important tokens
goal = reduce memory footprint under quality constraint
```

对我们的启发：

这些方法主要问：

```text
which tokens should remain resident?
```

我们更想问：

```text
given all or most tokens still exist somewhere,
which tokens should current q_t retrieve?
```

这两个问题相关但不同。Eviction 是 capacity policy；K graph 是 retrieval policy。

## 3. 系统领域给我们的统一抽象

综合以上工作，indexing 系统通常由五个部分组成：

```text
1. key space:
   what variable is searchable?

2. organization:
   tree, trie, postings list, graph, cluster, page table

3. candidate generation:
   how to avoid full scan?

4. verification / exact read:
   how to compute final score or read payload?

5. maintenance:
   how to insert, evict, compact, move, or update entries?
```

映射到 KV cache：

| 系统抽象 | KV-cache 版本 |
|---|---|
| key space | centered K, L2 K, pre-RoPE K, block summaries, anchor vectors |
| organization | paged blocks, radix prefix tree, K graph, anchors, clusters |
| candidate generation | local window, graph expansion, ANN search, learned indexer |
| exact read | exact qK over candidates + exact/residual V read |
| maintenance | streaming insert, block eviction, hot/cold tiering, graph update |

这也给出我们下一步最清晰的 research contract：

```text
physical prior:
  K-cache residual geometry can be used as an address index.

math model:
  build causal K graph under centered cosine / L2.

algorithm:
  generate candidates by local seeds + graph expansion.

experiment:
  compare candidates with full qK attention.

pass:
  graph beats local-only and random baselines under same candidate budget.

fail:
  graph does not improve attention mass recall or causes large CE delta.
```

## 4. 对我们当前课题的具体启发

### 4.1 不要把 KV cache 只当 tensor，要当 database object

现有系统已经说明 KV cache 有多重身份：

```text
tensor:
  K/V numeric values

memory pages:
  GPU blocks and page tables

prefix state:
  exact token sequence reuse

transport object:
  GPU/CPU/SSD/network movement

retrieval memory:
  q_t selects relevant historical K/V
```

我们的贡献点应聚焦最后一个：

```text
retrieval memory
```

也就是 attention-relevant K-side indexing。

### 4.2 应该做两阶段，而不是一步到位压缩

传统搜索系统不会直接把所有文档压成一个向量回答 query。它会：

```text
cheap recall -> exact score / rerank -> read payload
```

我们也应坚持：

```text
cheap K index -> candidate tokens / blocks -> exact qK on candidates -> V read
```

这和当前 `K-indexed, V-faithful` framing 完全一致。

### 4.3 图索引要和 layer/head 绑定

向量 DB 通常只有一个 embedding space。但 LLM KV cache 有：

```text
layer x head x token
```

Round2 已经显示不同 head 的 K graph 差异极大。因此系统设计不能写成：

```text
one global K graph
```

更合理的是：

```text
per-layer/head graph policy
some heads sparse
some heads local
some heads dense / exact
```

### 4.4 先做 attention recall，再谈 HNSW / tree / cluster

系统调研最容易诱导我们直接实现 HNSW、IVF、tree、cluster。但按 research-exploration 逻辑，这会过早。

现在最小可证伪问题是：

```text
K graph candidates 是否能 recall full qK attention？
```

如果不能，复杂 ANN index 只会把失败隐藏起来。

所以当前推荐顺序：

```text
1. local / random / K graph 1-hop / 2-hop recall
2. visualize good / bad examples
3. diagnose failure
4. only then choose HNSW-like, IVF-like, tree-like, or learned indexer
```

### 4.5 如果 inference-time graph 不够，需要训练时学 index

系统方法默认 index 是外部构建的。但 LLM 的 K-space 是模型内部表示，可能没有被训练成适合外部 sparse index。

如果 inference-time K graph recall 失败，可能需要转向：

```text
learned K-side indexer
query-to-anchor auxiliary loss
graph-aware sparse attention during training
CSA-like learned compressor/indexer
```

这时我们的研究就从：

```text
systems index over pretrained KV cache
```

变成：

```text
train model to produce indexable KV cache
```

## 5. 建议下一步实验

### 5.1 Query-Attention Recall Micro-Test

目标：

```text
验证 K graph candidates 是否服务 full qK attention。
```

Baselines：

```text
local-only
random same-size
high in-degree anchors
centered cosine graph 1-hop
L2 graph 1-hop
centered cosine graph 2-hop
local + graph
```

Metrics：

```text
attention_mass_recall
top_attention_token_recall
candidate_ratio
CE_delta
per-layer/head recall
```

Debug artifacts：

```text
token arc graph
candidate path
full attention top tokens
missed high-attention tokens
good / bad / confusing examples
```

### 5.2 System-Aware Candidate Cost

如果 recall 通过，下一步不要只看 candidate count，还要加入 cost：

```text
candidate_score = attention_relevance - fetch_cost
```

fetch cost 可以来自：

```text
same KV page
same GPU block
CPU offload
SSD / remote cache
```

这会把我们的算法和 vLLM / LMCache / CacheGen 这类系统工作接上。

### 5.3 Prefix Reuse 与 In-Sequence Retrieval 分开

报告中所有系统工作都说明：KV cache 有不同层次的问题。

建议后续文档中明确分开：

```text
cross-request exact reuse:
  RadixAttention / LMCache / CacheBlend style

within-sequence attention retrieval:
  our K graph / query-attention recall problem
```

不要把这两个问题混成一个“KV cache index”。

## 6. 和当前 Project Overview 的关系

本调研支持 `main_seq_compress/project_overview.md` 中的 current-state framework：

```text
K-indexed, V-faithful KV cache compression
```

更具体地说，系统领域给我们的启发是：

```text
K-indexed:
  index should be a lightweight candidate generator,
  not the final attention computation.

V-faithful:
  after candidate generation, payload should be read accurately,
  because search systems always separate index from content.

next falsification:
  candidate recall must be tested against full qK attention,
  not only against K-K geometry.
```

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437)
- [SGLang RadixAttention documentation](https://sgl-project-sglang-93.mintlify.app/concepts/radix-attention)
- [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665)
- [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion](https://arxiv.org/abs/2405.16444)
- [CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving](https://arxiv.org/abs/2310.07240)
- [KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving](https://arxiv.org/abs/2605.13734)
- [RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval](https://arxiv.org/abs/2409.10516)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management](https://arxiv.org/abs/2406.19707)
- [The Ubiquitous B-Tree](https://g-trees.github.io/g_trees/assets/references/comer1979ubiquitous.pdf)
- [The Log-Structured Merge-Tree](https://db.cs.berkeley.edu/cs286/papers/lsm-acta1996.pdf)
- [The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google/pubs/the-anatomy-of-a-large-scale-hypertextual-web-search-engine/)
- [From MaxScore to Block-Max WAND](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148045/)
- [Faiss: A library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [Efficient and robust approximate nearest neighbor search using HNSW](https://huggingface.co/papers/1603.09320)
- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://papers.nips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
