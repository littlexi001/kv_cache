# K-cache Graph Index Round4: Threshold Cluster Solution Findings

Date: 2026-06-04

## 0. 本轮核心结论

当前我们已经从“观察 K-cache 几何性质”推进到“基于这些性质设计 solution”。

本轮最重要结论是：

> **K-cache 的中后层几何结构支持一种 global, layer/head-specific, threshold-based cluster routing 方法。当前最有效的形态不是 pure cluster，而是 `local + global cluster threshold`：保留局部上下文，同时用 normalized L2 threshold 从全局 K-space 中选择相关 memory region，然后在选中 cluster 内全量做 exact qK attention 和 V read。**

在 quick setting 中，这个方法已经通过了原理验证：

```text
data: biomed_long_range_facts_hard_compact
seq_len: 3000
decode_start: 2500
num_clusters: 20
cluster root: kmeans center
distance scale: per-layer/per-kv-head prefill K-K sampled median L2
candidate policy: threshold select clusters, read all tokens inside selected clusters
```

关键结果：

| Method | Threshold | PPL | PPL ratio vs full | Candidate ratio | Approx compression | Attention mass |
|---|---:|---:|---:|---:|---:|---:|
| full | - | 1.797 | 1.00x | 100% | 1.0x | - |
| local_cluster_threshold | 0.50 | 3.001 | 1.67x | 9.0% | 11.1x | 0.759 |
| local_cluster_threshold | 0.75 | 1.925 | 1.07x | 20.3% | 4.9x | 0.817 |
| local_cluster_threshold | 1.00 | 1.815 | 1.01x | 38.1% | 2.6x | 0.872 |
| local_cluster_threshold | 1.25 | 1.855 | 1.03x | 48.4% | 2.1x | 0.898 |

从 KV-cache compression 角度，当前最合理的 Pareto 点不是质量最接近 full 的 `threshold=1.00`，而是：

> **`threshold=0.75`：只读约 20.3% token，约 4.9x 等效压缩，同时 PPL ratio 只有 1.07x。**

`threshold=1.00` 是 upper-quality point，但多读近一倍 token，只把 PPL 从 `1.925` 改善到 `1.815`，边际收益明显下降。

## 1. 我们已经理解的 K-cache 物理/数学性质

### 1.1 K 是 address-like，V 是 content-like

Round1 / Round2 的几何实验已经支持：

```text
K-cache 更各向异性、更局部平滑、更有 block / graph 结构；
V-cache 更 content-like，更需要高保真读取，不适合简单粗暴压缩。
```

因此当前总框架仍然是：

```text
K-indexed, V-faithful KV cache compression
```

也就是：

```text
K 侧负责便宜地找候选 memory region；
V 侧负责在候选上高保真读取内容。
```

### 1.2 K-space 适合建图/聚类，但不是所有层都一样

Round2 graph geometry 支持：

```text
K-space 中存在 local continuity + long-range shortcut 的多尺度图结构。
L2 和 centered cosine 都能观察到相近趋势。
不同 layer/head 差异很大。
```

Round3 / Round4 的 query-attention responsibility 进一步支持：

```text
浅层 L0 的 K cluster 对 future attention responsibility 预测较弱；
中后层 L13 / L27 的 attention mass 会明显集中在少数 K-space cluster 内。
```

典型结果：

| Layer | local mass | q_l2 nearest mass | cluster_l2 mass | oracle top2 cluster mass | effective clusters |
|---:|---:|---:|---:|---:|---:|
| L0 | 0.081 | 0.186 | 0.142 | 0.474 | 8.98 |
| L13 | 0.035 | 0.736 | 0.714 | 0.824 | 3.23 |
| L27 | 0.001 | 0.989 | 0.949 | 0.990 | 1.11 |

这说明：

> K-space cluster 不是普适地替代 attention；它在中后层更像有效的 memory index。

### 1.3 Top-k 不是物理模型，threshold 更符合当前理解

Top-k 隐含假设：

```text
每个 query 需要读取的历史 token / cluster 数量大致固定。
```

但这个假设不符合 long-context retrieval：

```text
有些 query 只需要少量历史信息；
有些 query 可能对应多个远程事实、多个重复片段、或多个相关 memory region。
```

因此我们现在采用的物理模型是：

```text
在固定 layer/head 的 K-space 中，query 应读取所有距离足够近的 memory region。
候选数量应由 query 的几何关系自适应决定，而不是由固定 top-k 决定。
```

## 2. 本轮 solution 设计

### 2.1 Falsifiable conjecture

> 在中后层 K-space 中，存在一个 layer/head-wise normalized L2 threshold，使得 query 通过该阈值选择出的 cluster block 能覆盖大部分 future attention responsibility；加上 local channel 后，可以在显著压缩候选 token 数的同时保持接近 full attention 的 decode loss。

这个 conjecture 可以被 falsify：

```text
如果 threshold 选出的 cluster mass 很低，则 K-space cluster 不是有效 index。
如果 mass 高但 PPL 差，则 candidate recall 不是主要瓶颈，可能是层间误差或 V/content 问题。
如果必须读接近全量 token 才能接近 full PPL，则方法没有压缩价值。
如果只在少数 query/layer/head 成立，则需要 layer/head-aware 或训练时约束。
```

### 2.2 Physical priors

本方法的每条策略都对应一个物理先验：

| Strategy | Physical prior |
|---|---|
| 对 prefill K 建 cluster | K-space 中存在 memory regions，不是随机散点 |
| layer/head-wise 建 cluster | 不同 layer/head 的 K 几何尺度和职责不同 |
| 用 median pairwise L2 做归一化尺度 | 每个 layer/head 有自己的 K-space 物理长度单位 |
| threshold 选 cluster | 相关 memory region 应由距离区域决定，而不是固定数量 |
| cluster 内全读 | cluster 是 coarse block index，block 内应做 exact scan |
| 加 local channel | 模型仍依赖近邻上下文、语法连续性和局部 residual 信息 |
| exact qK on candidates | K index 只负责 recall，不替代 attention scoring |
| V faithful read | V 是 content，不在本轮粗压缩 |

### 2.3 Mathematical model

对每个 layer `l` 和 KV head `h`，取 prefill K：

```text
K_prefill^{l,h} = {k_0, ..., k_{T_prefill-1}}
```

对 prefill K 做 clustering：

```text
C_1, ..., C_M
center_m = mean({k_i | i in C_m})
```

定义该 layer/head 的 K-space 距离尺度：

```text
scale_{l,h} = median_sampled_pairwise_L2(K_prefill^{l,h})
```

对 decode query `q_t`，计算归一化距离：

```text
d_norm(q_t, center_m) = ||q_t - center_m||_2 / scale_{l,h}
```

选择 cluster：

```text
SelectedClusters(q_t) = {m | d_norm(q_t, center_m) <= threshold}
```

若没有 cluster 通过阈值，则保底选择最近 cluster：

```text
min_selected_clusters = 1
```

候选 token：

```text
Candidates(q_t) =
  local_window(t)
  union sink_tokens
  union all tokens in SelectedClusters(q_t)
```

最终仍做 exact qK attention：

```text
Attention(q_t, K_candidates, V_candidates)
```

注意：

> cluster 内不再按 token-center 固定距离 threshold。否则离 center 远的 token 会永远选不到，这不是 query-dependent retrieval。

## 3. 实验设置与结果

### 3.1 输出位置

本轮 threshold quick 输出：

```text
fdong_seq_compress/outputs/cluster_threshold_quick_20260604_163720
```

主要表：

```text
aggregate_perplexity_by_method.csv
aggregate_timing_by_method_layer.csv
threshold_*/layer_stats_local_cluster_threshold.csv
threshold_*/layer_stats_cluster_threshold.csv
```

实验配置：

```text
model: Qwen3-0.6B
device: mps
dtype: float16
text: biomed_long_range_facts_hard_compact
seq_len: 3000
decode_start: 2500
num_eval_tokens: 500
num_clusters: 20
kmeans_steps: 3
cluster_scale_sample_pairs: 20000
max_candidates: 0  # 不截断，cluster 内全读
methods:
  cluster_threshold
  local_cluster_threshold
thresholds:
  0.25, 0.50, 0.75, 1.00, 1.25
```

### 3.2 现象 A：pure cluster threshold 失败，local channel 必须保留

Pure cluster threshold 的 PPL 非常差：

| Threshold | cluster_threshold PPL | Attention mass |
|---:|---:|---:|
| 0.25 | 4245.6 | 0.287 |
| 0.50 | 3652.7 | 0.287 |
| 0.75 | 1507.7 | 0.350 |
| 1.00 | 619.8 | 0.413 |
| 1.25 | 310.1 | 0.422 |

解释：

```text
仅靠 global cluster 无法替代完整上下文。
模型仍然强依赖 local channel，包括局部语法、短程依赖、sink/prefix 和 residual continuity。
```

因此当前 solution 必须是：

```text
local + global cluster threshold
```

而不是：

```text
global cluster only
```

### 3.3 现象 B：local + cluster threshold 成立，并形成清晰 Pareto 曲线

`local_cluster_threshold` 结果：

| Threshold | PPL | PPL ratio | Candidate ratio | Approx compression | Attention mass | Top10 recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 3.620 | 2.01x | 8.8% | 11.4x | 0.763 | 0.768 |
| 0.50 | 3.001 | 1.67x | 9.0% | 11.1x | 0.759 | 0.763 |
| 0.75 | 1.925 | 1.07x | 20.3% | 4.9x | 0.817 | 0.825 |
| 1.00 | 1.815 | 1.01x | 38.1% | 2.6x | 0.872 | 0.870 |
| 1.25 | 1.855 | 1.03x | 48.4% | 2.1x | 0.898 | 0.892 |

这个结果支持：

```text
threshold 控制了 quality-compression tradeoff；
候选数量随 query/threshold 自适应变化；
0.75 是当前最好的 compression-quality Pareto 点；
1.00 是接近 full-quality 的 upper-quality point。
```

### 3.4 现象 C：0.75 是当前更经济的阈值

从 0.50 到 0.75：

```text
PPL: 3.001 -> 1.925
candidate ratio: 9.0% -> 20.3%
```

质量大幅提升，代价可接受。

从 0.75 到 1.00：

```text
PPL: 1.925 -> 1.815
candidate ratio: 20.3% -> 38.1%
```

质量只小幅提升，但读取 token 接近翻倍。

因此当前工程/压缩角度的 sweet spot 是：

> **threshold = 0.75**

而不是单纯追求 PPL 最低的 `threshold=1.00`。

### 3.5 现象 D：阈值的分层行为不均匀

`local_cluster_threshold` 的部分层统计：

| Threshold | Layer | Candidate count | Selected clusters | Mass | Top10 recall |
|---:|---:|---:|---:|---:|---:|
| 0.75 | L0 | 272.0 | 1.00 | 0.770 | 0.875 |
| 0.75 | L13 | 602.3 | 3.34 | 0.786 | 0.850 |
| 0.75 | L20 | 719.6 | 4.85 | 0.874 | 0.781 |
| 0.75 | L27 | 264.5 | 1.00 | 0.952 | 0.946 |
| 1.00 | L13 | 2229.0 | 16.68 | 0.922 | 0.917 |
| 1.00 | L20 | 2101.4 | 15.78 | 0.958 | 0.884 |
| 1.00 | L27 | 264.6 | 1.00 | 0.953 | 0.946 |

解释：

```text
不同 layer 对阈值敏感程度不同。
L27 只需要很少 cluster 就能覆盖高 mass。
L13/L20 在 threshold=1.0 时会选非常多 cluster，代价显著上升。
```

这说明后续不一定应使用全层统一 threshold。更合理的方向是：

```text
layer/head-specific threshold
或只对部分 layer/head 启用 global cluster routing
```

## 4. 当前方法效果还不够好的地方

### 4.1 算法问题：pure cluster 缺 local channel 会崩

类型：

```text
算法设计问题
```

现象：

```text
cluster_threshold PPL 极差，即使 threshold=1.25 仍远离 full。
```

原因：

```text
global K cluster 只解决 long-range memory region recall；
它没有覆盖局部语言建模、短程依赖、prefix/sink 等模型实际使用的信息。
```

是否可解决：

```text
可解决。
当前已通过 local_cluster_threshold 解决第一版问题。
后续可进一步区分哪些 layer/head 需要 local，哪些需要 global。
```

### 4.2 算法问题：全层统一 threshold 不是最优

类型：

```text
算法校准问题
```

现象：

```text
L13/L20 在 threshold=1.0 时选中大量 cluster；
L27 在同一 threshold 下仍只需少数 cluster。
```

原因：

```text
不同 layer/head 的 K-space 几何分布和 retrieval responsibility 不同。
即使用 median pairwise L2 做尺度归一化，也不能保证一个全局 threshold 对所有 layer/head 都是 Pareto optimal。
```

是否可解决：

```text
可解决。
可用 per-layer/per-head threshold，或训练一个轻量 gate 来决定 threshold / budget。
```

### 4.3 算法问题：推理时方法没有训练一致性

类型：

```text
训练-推理 mismatch
```

现象：

```text
当前方法是在已经训练好的 dense attention 模型上，推理时强行改变 attention support。
即使 candidate mass 高，也可能因为模型没在这种稀疏 support 下训练过而产生层间误差累积。
```

当前证据：

```text
threshold=0.75 已经很接近 full，但仍有 PPL ratio 1.07x。
```

解释：

```text
这 7% PPL gap 不一定来自 candidate recall 不足；
也可能来自模型没有适应 threshold cluster attention 的分布。
```

是否可解决：

```text
原则上可解决。
更强方案应是训练时加入同样的 cluster-threshold attention / routing 约束，让模型学会在这种 support 上工作。
```

可能路线：

```text
1. 预训练/继续训练时使用 local + cluster-threshold sparse mask。
2. 对 K-space 加 regularization，让同一 retrieval region 的 K 更可聚类。
3. 训练一个 lightweight indexer / gate，预测 threshold 或 cluster selection。
4. 蒸馏 dense attention 到 sparse cluster attention。
```

### 4.4 工程问题：当前 cluster 构造和 mask 构造成本高

类型：

```text
工程实现问题
```

现象：

```text
当前 Python/CPU 侧 kmeans、candidate set 和 dense mask 构造很慢；
在之前 8000-token 实验中，cluster method 的 candidate/mask 构造占主要时间。
```

原因：

```text
当前代码是研究验证版：
  使用 dense qK + mask 来评估 PPL；
  cluster candidate 构造大量发生在 CPU/Python set/list；
  没有真正用 gather K/V 的 sparse attention kernel；
  没有 GPU/MPS optimized clustering/index。
```

是否可解决：

```text
工程上可解决，但需要专门实现。
这不是算法原理不可行，而是当前原型没有做系统优化。
```

可能路线：

```text
1. prefill 阶段在 GPU/MPS 上批量计算 K-center 距离。
2. 使用 tensorized membership mask，而不是 Python set。
3. 用 gather 选 K/V 后做小矩阵 exact attention，避免 dense T x T mask。
4. 离线/分块更新 cluster，避免每步重新构造。
5. 真正部署时只在 selected cluster 上读 V，避免 dense attention score。
```

### 4.5 硬件问题：full attention matmul 被高度优化，稀疏 routing 未必立刻更快

类型：

```text
硬件/系统问题
```

现象：

```text
GPU/MPS 对大矩阵乘法高度优化；
naive sparse routing 可能因为不连续 gather、CPU 调度、mask 构造而慢于 full attention。
```

是否可解决：

```text
部分可解决。
算法上有压缩空间，但要转化成真实 speedup，需要硬件友好的 block-level layout 和 kernel。
```

工程启发：

```text
cluster 不应只是数学概念；
它也应对应连续内存 block / page / KV block。
否则即使候选 token 少，实际系统也可能不快。
```

## 5. 当前方法的 claim boundary

### 已经可以说

```text
1. Qwen3-0.6B 的中后层 K-cache 确实具有可索引的 cluster responsibility 结构。
2. 基于 layer/head-wise normalized L2 threshold 的 cluster selection 比固定 top-k 更符合 compression-quality tradeoff。
3. local + global cluster threshold 在 quick hard long-range setting 中可以用约 20.3% token 达到 PPL ratio 1.07x。
4. 该方法的核心原理已经通过：K-space threshold 可以作为 memory region router。
```

### 还不能说

```text
1. 该方法已经能真实加速推理。
2. 该方法在所有数据、所有模型、所有上下文长度上稳定。
3. 统一 threshold 是最终最优策略。
4. 推理时 cluster threshold 是最终形式，训练时不需要配合。
```

## 6. 下一步如何让方法变得更好

### 6.1 从推理时方法升级为训练时方法

这是当前最有研究价值的方向。

当前推理时方法的问题是：

```text
模型训练时看到的是 dense attention；
推理时突然只给 local + threshold cluster support；
模型没有被训练成主动把 K-space 组织成可阈值检索的 memory regions。
```

训练时方法可以让模型学会：

```text
哪些 token 应该落在同一个 K cluster；
哪些 layer/head 应承担 global retrieval；
如何让 query-center threshold selection 保持稳定；
如何在 sparse support 上保持 V read 的高保真。
```

可行设计：

```text
1. Sparse attention training:
   训练时直接使用 local + cluster-threshold attention mask。

2. Dense-to-sparse distillation:
   以 dense attention 的 output/logits 为 teacher，让 sparse cluster attention 学生拟合。

3. K-geometry regularization:
   鼓励承担相似 future responsibility 的 token 在 K-space 中更接近。

4. Learnable threshold/indexer:
   不手工固定 threshold，而是让模型预测 per-layer/head/per-query threshold 或 selected clusters。
```

这会把当前方法从：

```text
post-hoc inference-time compression
```

推进到：

```text
train-time index-aware KV-cache architecture
```

### 6.2 从 token cluster 变成 block/page cluster

为了工程可部署，cluster 应该对应 KV block：

```text
cluster = contiguous or page-like KV block
```

而不是任意散点 token set。

否则：

```text
候选 token 少，但 memory access 不连续；
GPU/MPS kernel 难以高效利用；
实际 latency 未必下降。
```

### 6.3 从统一 threshold 变成 per-layer/head threshold

当前结果显示：

```text
L27 可以用很少 cluster 得到高 mass；
L13/L20 对 threshold 更敏感，容易读太多。
```

因此下一步可以学习/搜索：

```text
threshold_{layer, head}
```

或策略：

```text
shallow layers: dense/local
middle layers: threshold cluster
deep layers: aggressive cluster compression
```

### 6.4 从 kmeans center 变成更适合 K-space 的 index root

当前 center 是 kmeans mean，不一定是最佳 root。

可替代：

```text
medoid root: cluster 内离 center 最近的真实 token
learned root: 训练得到的 index vector
hierarchical root: coarse-to-fine tree root
attention-responsibility root: 最大化 future attention mass recall 的 anchor
```

## 7. 更新后的研究链条

当前完整链条可以写成：

```text
Conjecture:
  LLM 的 K-cache 可以作为 address-like memory index，V-cache 应高保真读取。

Physical priors:
  K-space 中后层具有 graph/cluster-friendly structure；
  long-range relation 不应依赖 local window；
  query 所需 memory region 数量是可变的。

Mathematical model:
  在每个 layer/head 的 K-space 中，用 median pairwise L2 定义物理距离单位；
  query 通过 normalized L2 threshold 选择相关 cluster block；
  selected block 内 exact qK attention 负责精确读取。

Implementation contract:
  prefill K -> cluster;
  query -> threshold select cluster;
  cluster 内 token 全读;
  local channel 永远保留;
  V faithful read;
  输出 PPL / attention mass / candidate ratio / timing。

Experiment:
  threshold sweep 0.25-1.25;
  quick hard long-range data;
  compare pure cluster vs local+cluster.

Failure analysis:
  pure cluster fails -> local channel necessary;
  threshold=0.75 is Pareto sweet spot;
  threshold=1.0 quality better but compression worse;
  current runtime slow -> engineering prototype issue, not algorithm proof of impossibility;
  remaining PPL gap -> likely train/inference mismatch and layer/head threshold mismatch.

Updated conjecture:
  Local + layer/head-normalized cluster-threshold routing is a plausible KV-cache compression mechanism.
  The next substantial research direction should be train-time index-aware attention, not more blind inference-time sweep.
```

