# KV Cache 压缩课题 Current-State Framework

Date: 2026-06-01

## 0. 文档定位

这是 KV cache compression 课题的 current-state document。

它不是实验流水账，而是当前最干净的研究状态：

```text
conjecture
physical priors
mathematical models
implementation contracts
evidence boundary
next falsification tests
```

历史实验记录保留在 round-level iteration ledger：

```text
fdong_seq_compress/k_cache_graph_round1_findings.md
fdong_seq_compress/k_cache_graph_round2_handoff.md
```

## 1. 目标和当前结论

下游目标是：

> 在长上下文推理中，避免每个 query 都和全部历史 K 做 full qK score，同时尽量保留 next-token prediction 所需的 V-side 信息。

当前研究结论是：

> Qwen3-0.6B 的 K-cache 已经表现出足够稳定的 residual graph geometry，值得继续研究 K-side graph indexing；但我们还没有证明 K-K graph candidates 能召回 full qK attention 真正使用的 token。

所以当前状态是：

```text
已经支持：
  K-cache 有 graph-friendly geometry。

尚未证明：
  K-cache graph candidates 可以替代或近似 full qK retrieval。

下一道 gate：
  query-attention recall。
```

这个课题当前应被理解为：

```text
K-indexed, V-faithful KV cache compression
```

而不是 generic KV averaging。

## 2. 可证伪 Conjecture

当前 conjecture：

> LLM K-cache 不是无结构的 flat memory。去掉 common direction 后，它形成了稀疏、多尺度的 address space。如果这个 address-space graph 能召回 full qK attention 的重要 token，那么长上下文 attention 可以被拆成 cheap K-side candidate retrieval 和 high-fidelity V-side reading。

这个 conjecture 可以被以下结果证伪或削弱：

```text
1. K-K graph structure 存在，但不能 recall qK attention。
2. K-K graph 只优于 random，但不能优于 local-window baseline。
3. K-K graph recall 需要太大的 candidate set，无法节省计算。
4. graph-friendly structure 只存在于少数 synthetic text。
5. inference-time graph retrieval 失败，因为 pretrained K-space 没有被训练成 sparse retrieval index。
```

注意：一个 operationalization 失败，不等于整个方向失败。例如 inference-time K graph recall 失败后，仍可能有一个更新后的 conjecture：

```text
K-space 需要 training-time graph-aware constraint。
```

## 3. Physical Priors

### 3.1 Attention 是 Memory Retrieval

Physical prior：

> Attention 本质上是 memory retrieval：query vector 搜索历史 key vectors，并读取对应 value vectors。

数学模型：

```text
score_i = q_t · k_i
attn_i = softmax(score_i)
output_t = sum_i attn_i v_i
```

实现启发：

```text
不要只研究如何压小每个向量。
更关键的是如何减少需要访问的历史 positions / blocks。
```

当前证据：

```text
Round1 和 Round2 都把 K-cache 当成可能支持 retrieval 的 address space 来研究。
```

边界：

```text
retrieval 视角只提供 indexing 动机；
它本身不证明 K-K graph neighbors 对 qK retrieval 有用。
```

### 3.2 K 是 Index，V 是 Payload

Physical prior：

> K-cache 更像 address / routing / index；V-cache 更像 content / evidence / payload。

数学模型：

```text
K side:
  build summaries, anchors, graph edges, or block candidates

V side:
  preserve exact or residual value content and gather it after K-side selection
```

Round1 证据：

| 性质 | K-cache | V-cache |
|---|---|---|
| 有效维度 | 随 sequence length sublinear 增长 | 也 sublinear，但整体高于 K |
| 各向异性 | strong common direction | common direction 弱得多 |
| 局部平滑性 | 强 | 弱 |
| 小 block 结构 | 4/8/16 token 下明显 | 不适合直接平均 |
| 角色 | address / index / routing | content / evidence / payload |

实现启发：

```text
优先尝试：
  K index -> candidate positions / blocks -> exact or residual V read

不作为第一版方案：
  把 K 和 V 一起平均成 coarse memory vector
```

边界：

```text
这个 prior 支持 K-side compression 多于 V-side compression；
但还没有决定哪一种 K index 最好。
```

### 3.3 K 有 Residual Graph Geometry

Physical prior：

> 去掉 common direction 后，K vectors 形成有意义的 residual address space，其中同时存在 local continuity、long-range shortcuts 和 head-specific graph structure。

数学模型：

```text
centered K:
  r_i = k_i - mean_j(k_j)

cosine graph:
  edge i -> j if cos(r_i, r_j) is in causal top-k, j < i

L2 graph:
  edge i -> j if ||k_i - k_j||_2 is in causal top-k, j < i
```

为什么 L2 重要：

```text
|q · k1 - q · k2| <= ||q|| ||k1 - k2||
```

所以 K-K L2 distance 小，对 qK score stability 有直接解释。

Round2 证据：

```text
centered cosine 和 L2 都显示：
  local edges
  long-range edges
  neighbor-distance 随 seq_len sublinear 增长
  strong layer/head heterogeneity
  moderate hub / anchor structure
```

边界：

```text
这证明 K-cache 有 graph-friendly geometry；
但还没有证明 graph candidates 能 recall full qK attention。
```

### 3.4 可能需要 Training-Time Alignment

Physical prior：

> pretrained model 可能自然存在 graph-like K geometry，但并没有被训练成用这个 graph 做 sparse retrieval。

两条可能路线：

```text
Route A:
  pretrained K-space 已经足够好
  -> inference-time graph index 可以工作

Route B:
  pretrained K-space 只提供启发
  -> 需要 train-time graph-aware / index-aware constraint
```

边界：

```text
不要在 inference-time recall 被具体证伪之前，过早跳到新的 trainable architecture。
```

## 4. 当前数学对象

### 4.1 Cache Tensors

对 layer `l`、head `h`、token position `i`：

```text
k_{l,h,i} in R^{d_h}
v_{l,h,i} in R^{d_h}
```

token-level concatenated K：

```text
k_{l,i} = concat_h k_{l,h,i}
```

### 4.2 K Transform

默认 graph space：

```text
r_i = k_i - mean_j(k_j)
```

原因：

```text
k_i = c + r_i
q · k_i = q · c + q · r_i
```

对同一个 query，`q · c` 对所有历史 token 是同一个 additive constant，会被 softmax 抵消。真正产生 token 间选择性的部分是 `q · r_i`。

### 4.3 Graph Construction

Causal top-k K graph：

```text
G = (nodes, edges)
nodes = token positions
edge i -> j if j < i and k_j is among i's top-k nearest previous K nodes
```

支持的 metric：

```text
centered cosine
L2 distance
dot product only as diagnostic
```

### 4.4 Query-Attention Recall

这是下一阶段必须建立的对象。

Full attention oracle：

```text
a_t = softmax(q_t K_{<t}^T)
```

graph candidate set：

```text
C_t = local_window(t) union graph_expand(seed_nodes, hops)
```

Recall metrics：

```text
attention_mass_recall = sum_{i in C_t} a_{t,i}
top_token_recall@r = |top_r(a_t) intersect C_t| / r
candidate_ratio = |C_t| / t
CE_delta = CE(masked_candidates_only) - CE(full_attention)
```

## 5. Implementation Contracts

### 5.1 已完成：K/V Geometry Diagnosis

问题：

```text
K 和 V 是否具有不同的 compressibility 和 geometry？
```

输入：

```text
Qwen3-0.6B
long English text
prefix lengths from short to long
K/V cache tensors by layer/head
```

输出：

```text
effective rank
stable rank
anisotropy
local smoothness
block structure
subspace stability
new-token residual novelty
```

结果：

```text
K 更像 index。
V 更像 payload。
```

ledger：

```text
fdong_seq_compress/k_cache_graph_round1_findings.md
```

### 5.2 已完成：K Graph Geometry Sweep

问题：

```text
在不同 metric、sequence length、layer/head、text domain 下，K-cache 是否都有 graph-friendly structure？
```

输入：

```text
Qwen3-0.6B
5 synthetic long text domains
seq_len = 1000, 2000, 4000, 8000
similarity = centered cosine, L2
top_k = 10, 20, 50
all layers / all KV heads for layer-head sweep
```

输出：

```text
neighbor distance distribution
local edge fraction
long-range edge fraction
in-degree / hubness
layer/head heterogeneity
domain robustness
```

结果：

```text
K graph geometry 已经足够稳定，值得进入 query-attention recall tests。
```

ledger：

```text
fdong_seq_compress/k_cache_graph_round2_handoff.md
```

### 5.3 下一步：Query-Attention Recall

问题：

```text
K graph candidates 能否覆盖 full qK attention？
```

Algorithm：

```text
1. Select one text, one layer/head, and one prefix length.
2. Compute full qK attention for query positions.
3. Build causal K-K graph from historical K.
4. Generate candidates with local window, 1-hop graph, and 2-hop graph.
5. Compare candidates against full qK attention.
6. Save aggregate metrics and representative pass/fail examples.
```

Pass condition：

```text
Graph candidates 在同等 candidate budget 下明显优于 local-only 和 random same-size baseline。
```

Fail conditions：

```text
Graph candidates 不优于 local-only。
Graph candidates 需要太多 tokens 才有效。
Graph candidates 只在少数 head 上有效，且没有清晰模式。
Graph candidates recall top tokens 但漏掉 attention mass。
Graph candidates recall attention mass 但 CE delta 很大。
```

Debug artifacts：

```text
per-query candidate list
full attention top tokens
candidate overlap table
attention mass recall table
edge-path visualization for selected examples
good / bad / confusing examples
```

## 6. 当前证据

### 6.1 Round1 Evidence

Round1 建立了：

```text
K 和 V 的几何结构不同。
raw K similarity 会被 common direction 误导。
centered / residual K 更接近 attention-relevant geometry。
K residual space 有初步 nearest-neighbor 和 nonlocal edge structure。
```

Round1 最重要结论：

```text
K-indexed, V-faithful compression 比 joint KV averaging 更合理。
```

### 6.2 Round2 Evidence

Round2 建立了：

```text
K graph structure 在 L2 下也成立，不只是 centered cosine artifact。
K graph structure 随 seq_len 增长到 8000 时没有崩坏。
K graph 同时有 local edges 和 long-range shortcuts。
layer/head 差异很大，应该利用而不是平均掉。
主要现象在 5 类文本 domain 上都存在。
```

Round2 最重要结论：

```text
K-cache 在几何上适合继续探索 graph index；
但 attention-level usefulness 仍未验证。
```

## 7. 当前 Claim Boundary

现在可以说：

```text
Qwen3-0.6B K-cache has graph-friendly residual geometry.
K 和 V 应当被区别对待。
K-side indexing with V-side high-fidelity reading 是有实验证据支持的方向。
```

现在还不能说：

```text
K graph index 可以无损降低 attention compute。
K-K nearest neighbors 等价于 qK attention neighbors。
inference-time graph index 已经足够。
一定需要新的 trainable architecture。
```

## 8. 下一轮 Research Loop

下一轮按以下顺序做。

### 8.1 先做 Micro-Test

先跑一个小的 query-attention recall test，而不是直接大 sweep。

建议设置：

```text
text: long_textbook_distributed_systems or long_codebase_query_engine
model: Qwen3-0.6B
seq_len: 1000 or 2000
layers/heads:
  local head: L27/H0
  long-range head: L6/H3 or L15/H1
  mixed head: one median-distance head
candidate methods:
  local-only
  random same-size
  K graph 1-hop
  K graph 2-hop
  local + graph
metrics:
  attention mass recall
  top-token recall
  candidate ratio
```

### 8.2 必须有 Visual Evidence

每个 representative head 至少保存：

```text
token-position arc graph
node size by in-degree
edge color by distance or metric
selected query examples with full attention top tokens
candidate paths from graph expansion
```

Aggregate metrics 不够。我们需要看到 good / bad / confusing cases。

### 8.3 失败后先分解，不要马上改方案

如果 recall 失败，先 decomposes：

```text
big failure:
  graph candidates miss full qK attention

candidate causes:
  seed selection is wrong
  K-K edges are not query-relevant
  chosen layer/head is wrong
  metric is wrong
  local baseline is already enough
  attention target is diffuse and not top-token-like
  pretrained K-space is not trained for sparse graph retrieval
```

每个原因都要有 stage-local test，再决定是否改算法或上训练约束。

### 8.4 Recall 后的决策

如果 graph recall 通过：

```text
build minimal inference-time graph candidate attention
measure CE delta and runtime/candidate budget
```

如果 graph recall 失败，并且失败原因指向 train/inference mismatch：

```text
design training-time graph-aware K-space objective
for example:
  graph-aware attention mask
  query-to-anchor auxiliary loss
  learned CSA-like lightweight indexer
  K-side index with V-side faithful payload
```

## 9. Research Hygiene

后续每个实验都要记录：

```text
conjecture
physical prior
mathematical model
algorithm specification
input and hyperparameters
pass/fail criteria
stage-level artifacts
result
failure interpretation
updated conjecture
```

不要把 plausible mechanism 当成科研进展。进展只来自：

```text
conjecture survived a concrete falsification test
or
conjecture failed in a way that localized the wrong assumption
```
