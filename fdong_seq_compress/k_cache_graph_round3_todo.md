# K-cache Graph Index Round3 TODO

Date: 2026-06-03

## 0. Round3 当前问题

Round1 / Round2 已经支持：

```text
K-cache has graph-friendly residual geometry.
K-side indexing is worth studying.
```

Round3 需要往两个方向推进：

```text
1. 验证更强模型 / 更难长程数据下，是否真的出现显著 long-range high-attention score。
2. 在当前 Qwen3-0.6B 设置下，继续做 inference-time candidate method：graph / cluster center / anchor token 是否能提高 attention recall 或实际推理效果。
```

这两个方向回答不同问题，不能混在一起。

## 1. 老板提出的 TODO：更强模型 + 更难长程数据

### 1.1 核心问题

当前 synthetic text 可能不够难，也不一定真的要求模型使用远距离信息。因此，即使 K graph 有 long-range edges，也未必会在 qK attention 里出现显著远距离 high-attention score。

老板的判断是：

> 只要数据真的需要长程依赖，模型就应该出现显著 long-range high-attention score。

另一个同学的判断是：

> 不一定。可能要足够强的大模型，才能识别并使用这种远距离信息；小模型即使数据里有长程依赖，也未必 attend 到正确位置。

所以这里有两个待区分的因素：

```text
data difficulty / long-range dependency strength
model capability
```

### 1.2 建议构造的数据

需要自己造真正含有长程信息的数据，而不是普通长文章。

示例模式：

```text
position 1000:
  Sam is 3 years old.

position 20000:
  Question: How old is Sam?
```

可以扩展成多种 controlled long-range tasks：

```text
single fact recall:
  Sam is 3. Later ask Sam's age.

multi-entity recall:
  Sam is 3, Alice is 7, Bob is 12. Later ask one entity.

attribute binding:
  Sam's age / city / color / code are separated.

needle-in-haystack:
  one key fact buried in distractors.

conflict resolution:
  early fact and later update both exist, ask latest / earliest.

multi-hop:
  Sam's teacher is Alice. Alice's room number is 204. Ask Sam's teacher's room.
```

### 1.3 要比较的模型

建议至少比较：

```text
small model:
  Qwen3-0.6B

stronger model:
  Qwen3 larger variant or another locally available stronger LLM
```

核心不是马上追求大 sweep，而是先回答：

```text
同一条 long-range controlled data，
小模型和大模型的 qK attention 是否都能打到正确远程位置？
```

### 1.4 指标

对每个 query 位置，记录：

```text
answer correctness
full qK attention top positions
attention mass on ground-truth evidence span
rank of ground-truth evidence span
distance of top-attended tokens
whether position-0 / prefix sink dominates
```

关键判断：

```text
如果 harder data + stronger model 出现 high long-range attention：
  说明我们之前没看到强 qK long-range，可能是数据/模型不够。

如果 stronger model 也不 attend 远程 evidence：
  说明模型可能通过 residual state / MLP / distributed memory 解决，attention map 未必直接显式指向 evidence。

如果小模型不 attend，大模型 attend：
  支持“模型能力是关键因素”。

如果小模型和大模型都 attend：
  支持“数据难度是关键因素”。
```

### 1.5 归属

这个方向可能由合作者继续做。

我们需要之后接收他们的结果，并把它接入当前 framework：

```text
long-range task result
-> 是否存在 long-range high qK attention
-> 是否需要 K graph candidate recall
```

## 2. 我们当前 TODO：真正试 inference-time candidate method

### 2.1 当前已经做了什么

我们已经实现了第一版后验 attention recall test：

```text
full qK attention as oracle
local baseline
random baseline
local_topq_graph
local_graph_all
always include prefix/sink positions 0:10
```

当前观察：

```text
1. prefix/sink token 必须默认进入 candidate。
2. layer 27 local head 基本被 local window 覆盖，graph 帮助小。
3. layer 6 / layer 15 的部分 head 上，local_topq_graph 明显优于 local。
4. graph 方法不是所有 head 都有效，应当 layer/head-aware。
```

需要注意：

```text
local_graph_all 之前有一个截断 bug，已修复。
需要重跑后再解释 local_graph_all。
```

### 2.2 下一步最小实验

先重跑修复后的 layer subset：

```text
LAYERS=6,15,27
Q_HEADS=all
MAX_TOKENS=2000
DECODE_START=1000
ALWAYS_INCLUDE_POSITIONS=0:10
```

重点看：

```text
local_topq_graph vs local
local_graph_all vs local
random_max_candidates vs local
per-layer/head improvement
```

重点 heads：

```text
L6 QH6
L6 QH7
L6 QH10
L6 QH13
L15 QH6
L15 QH8
```

### 2.3 真正推理实验

attention recall 只是后验上限。下一步需要试真实推理：

```text
full attention baseline
candidate-only attention
compare logits / CE / generated answer
```

最小版本可以先不追求工程速度：

```text
1. 选固定 layer/head。
2. 用 candidate mask 限制 qK attention。
3. 计算 masked attention output。
4. 比较 full attention output / logits / CE delta。
```

难点：

```text
Qwen3 attention 是多层多头耦合。
只 mask 单层单头可能很难看到 end-to-end generation effect。
全层全头 mask 又需要较多工程改动。
```

因此可以先做 stage-local output error：

```text
per-head attention output error:
  ||o_candidate - o_full|| / ||o_full||

per-layer attention output error:
  aggregate heads after candidate masking

then:
  CE / logits delta
```

### 2.4 Cluster center / anchor token 方法

老板提到的 graph / cluster / anchor 可以拆成几个 candidate generator。

#### 方法 A：High in-degree anchor

构造：

```text
1. 对 K-K graph 计算 in-degree。
2. 选高 in-degree nodes 作为 anchors。
3. query 先和 anchors 做 qK。
4. 选 top anchor。
5. 展开 anchor neighborhood。
```

要测：

```text
anchor attention recall
anchor neighborhood recall
candidate size
是否优于 local + random
```

#### 方法 B：K-medoids / cluster center

构造：

```text
1. 对历史 K 做 clustering。
2. 每个 cluster 选 medoid / center。
3. query 先 score centers。
4. 选 top clusters。
5. 在 selected clusters 内做 exact qK。
```

注意：

```text
K-means center 可能不是真实 token，不能直接读 V。
medoid 是真实 token，更容易和 KV cache 对齐。
```

要测：

```text
center recall
selected-cluster attention mass recall
cluster size distribution
long-range evidence 是否被同 cluster 收进来
```

#### 方法 C：Local seed + graph expansion

当前已有：

```text
local window
-> top qK local seeds
-> K graph 1-hop / 2-hop expansion
```

这个方法的特点是：

```text
不需要全局 cluster。
更像 query-time graph traversal。
```

#### 方法 D：Local + anchor hybrid

可能更稳：

```text
candidate = prefix/sink + local window + selected anchor neighborhoods
```

因为 prefix/sink 和 recent local 基本是必保留路径。

### 2.5 工程收益暂时不是第一优先级

当前阶段先回答算法上限：

```text
candidate recall 能不能高？
candidate size 能不能小？
哪些 layer/head 需要 graph？
```

暂时不纠结：

```text
CPU top-k 慢
cluster update 慢
non-contiguous gather 慢
kernel 不好写
```

这些是后续 system optimization 问题。当前先找：

```text
存在性证据 / upper bound
```

## 3. Round3 当前最小计划

建议明天从这三件事开始：

```text
1. 重跑修复后的 attention recall subset。
2. 汇总哪些 layer/head graph 真正超过 local。
3. 设计第一个 anchor / cluster center candidate generator。
```

如果时间够，再写：

```text
k_graph_attention_recall_plan.md
```

把每个 candidate generator 的：

```text
input
parameters
algorithm
pass condition
fail condition
debug artifacts
```

写清楚后再继续加代码。
