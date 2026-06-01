# K-cache Graph Index Round2 Findings

Date: 2026-06-01

## 0. Round2 完整实验后的核心结论

当前核心问题是：

> Qwen3-0.6B 的 K-cache 是否适合构建图索引，从而让 query 不必和所有历史 K 做 full qK score？

Round2 完整 sweep 后的判断是：

> **K-cache 值得继续探索图索引。更准确地说，它适合作为 K-side candidate graph / routing index 的候选空间；但还没有证明它可以直接替代 full attention。**

这个结论的边界很重要：

```text
已经支持：
K-cache 自身存在稳定、跨文本、跨 metric 的图结构。

尚未证明：
K-K graph candidates 能高 recall 覆盖真实 qK attention top tokens / attention mass。
```

因此 Round2 的结论不是“我们已经有一个可部署的 graph attention solution”，而是：

> K-cache 的几何结构已经足够不像随机噪声，值得进入 query-attention recall 和推理时 candidate-generation 验证。

### 0.1 本轮实验配置

本轮完整输出目录：

```text
fdong_seq_compress/outputs/round2_geometry_sweeps_20260601_195004
```

实验状态：

```text
100 summary.json
5 个 domain 文本
每个 domain 20 个实验 summary
完成时间：2026-06-01 20:37:52 CST
```

五个文本域：

```text
long_english_12000_words
long_codebase_query_engine
long_textbook_distributed_systems
long_news_supply_chain_dossier
long_dialogue_tool_transcript
```

三类 sweep：

```text
1. L2 metric sweep
2. seq-len scaling sweep
3. layer/head selection sweep
```

这些 sweep 分别回答不同问题：

| Sweep | 变化的超参 | 回答的问题 | 结论 |
|---|---|---|---|
| L2 metric sweep | `top_k=10/20/50`, `analysis_level=token/head`, `similarity=l2` | K-K graph 是否在 score-stability metric 下也成立 | 成立。L2 和 centered cosine 给出相近的 neighbor distance / hubness 结构 |
| Seq-len scaling sweep | `seq_len=1000/2000/4000/8000`, `similarity=cos/l2` | 图结构随上下文变长是否崩坏 | 没有崩坏。近邻距离随长度增长，但 sublinear；长程边比例上升 |
| Layer/head selection sweep | `layers=all`, `heads=all`, `top_k=10/20/50`, `similarity=cos/l2` | 是否所有层/头都同质，还是存在图结构分工 | 不同 layer/head 差异很大；有些 head 极局部，有些 head 明显长程 |
| Domain sweep | 5 类长文本 | 现象是否只是某一种 synthetic text artifact | 主要现象跨 domain 稳定存在 |

### 0.2 直接支持结论的主要现象

#### 现象 A：K-cache 近邻不是纯局部窗口，但也不是随机远距离

在 seq-len scaling sweep 中，centered cosine top-10 的平均统计为：

| seq_len | p50 distance | p90 distance | frac <= 8 | frac <= 32 | frac >= 128 | frac >= 256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 38.4 | 350.4 | 0.311 | 0.477 | 0.309 | 0.143 |
| 2000 | 63.0 | 690.4 | 0.290 | 0.426 | 0.402 | 0.258 |
| 4000 | 96.4 | 1247.4 | 0.276 | 0.392 | 0.466 | 0.345 |
| 8000 | 132.4 | 1960.0 | 0.268 | 0.372 | 0.503 | 0.396 |

这说明：

1. `seq_len` 从 1000 增长到 8000，近邻 median distance 只从 `38.4` 增长到 `132.4`，不是线性变成 8 倍。
2. 8000 token 时仍有 `26.8%` top-10 边落在距离 `<=8`，说明局部平滑结构稳定存在。
3. 8000 token 时也有 `50.3%` top-10 边跨越 `>=128`，说明 K graph 不是 local window 的同义词。

这支持：

> K-cache 有多尺度图结构：local continuity + long-range shortcut 同时存在。

#### 现象 B：L2 和 cosine 给出一致趋势，说明不是单个 metric artifact

同样的 seq-len scaling sweep 中，L2 top-10 的平均统计为：

| seq_len | p50 distance | p90 distance | frac <= 8 | frac <= 32 | frac >= 128 | frac >= 256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 40.0 | 348.2 | 0.300 | 0.468 | 0.317 | 0.144 |
| 2000 | 69.4 | 694.0 | 0.280 | 0.416 | 0.412 | 0.262 |
| 4000 | 107.6 | 1245.6 | 0.265 | 0.380 | 0.479 | 0.351 |
| 8000 | 142.0 | 1930.8 | 0.258 | 0.361 | 0.515 | 0.401 |

L2 与 centered cosine 的结论基本一致：

```text
local fraction 稳定存在
long-range fraction 随 seq_len 增长上升
neighbor distance sublinear 增长
```

这很关键，因为 L2 有直接的 qK score-stability 含义：

```text
|q · k1 - q · k2| <= ||q|| ||k1 - k2||
```

所以 L2 sweep 支持：

> K-cache 的图结构不只是 cosine 方向相似的偶然现象，也与 qK score 稳定性有一定几何关联。

#### 现象 C：top-k budget 控制局部/长程混合比例

在 1000-token、head-level、全 layer/head 的 stage3 sweep 中：

| metric | top_k | p50 distance | frac <= 8 | frac <= 32 | frac >= 128 | frac >= 256 |
|---|---:|---:|---:|---:|---:|---:|
| cosine | 10 | 29.0 | 0.316 | 0.520 | 0.313 | 0.129 |
| cosine | 20 | 52.4 | 0.207 | 0.430 | 0.360 | 0.160 |
| cosine | 50 | 93.2 | 0.109 | 0.294 | 0.437 | 0.212 |
| L2 | 10 | 29.0 | 0.311 | 0.520 | 0.310 | 0.126 |
| L2 | 20 | 51.6 | 0.204 | 0.429 | 0.355 | 0.157 |
| L2 | 50 | 91.8 | 0.107 | 0.290 | 0.433 | 0.209 |

这说明：

> candidate budget 本身就是控制 graph 语义的超参。小 top-k 更像局部/强连接，大 top-k 会自然引入更多 long-range shortcut。

这对后续系统设计有直接含义：

```text
top_k 小：更适合便宜的 local-like candidate
top_k 大：更适合 recall long-range evidence
```

#### 现象 D：layer/head 差异很大，K graph 不应一刀切

以 1000-token、centered cosine、top-10、全 domain 平均为例：

最局部的 heads：

| layer/head | p50 distance | frac <= 8 | frac >= 128 | frac >= 256 |
|---|---:|---:|---:|---:|
| L27/H0 | 6.0 | 0.721 | 0.037 | 0.033 |
| L17/H3 | 6.0 | 0.719 | 0.126 | 0.026 |
| L27/H1 | 6.0 | 0.693 | 0.080 | 0.075 |
| L11/H7 | 6.0 | 0.669 | 0.019 | 0.009 |

最明显长程的 heads：

| layer/head | p50 distance | frac <= 8 | frac >= 128 | frac >= 256 |
|---|---:|---:|---:|---:|
| L6/H3 | 184.0 | 0.077 | 0.623 | 0.364 |
| L15/H1 | 176.0 | 0.029 | 0.602 | 0.378 |
| L11/H1 | 173.0 | 0.128 | 0.589 | 0.320 |
| L3/H5 | 170.8 | 0.104 | 0.579 | 0.325 |

这说明：

> K-cache graph 的结构是 layer/head-specific 的。后续不能默认所有 head 用同一种图索引策略。

合理路线应当是：

```text
先找 graph-friendly heads
再做 per-head candidate recall
最后决定哪些 layer/head 值得建图、哪些保留 dense/local
```

#### 现象 E：hub / anchor 存在，但不是极端单点支配

本轮 top-10 设置下，top 1% 节点大约吸收 `4%~5%` 入边；top-k 增大后，max indegree 明显升高。

这说明：

```text
存在 anchor / hub token
但不是所有边都坍缩到极少数 token
```

这对图索引是正面信号：anchor 可以成为 routing / coarse index 的候选，但 hubness 还没有极端到让 graph 失去区分度。

### 0.3 当前结论的限制

Round2 仍然只是几何实验。它回答的是：

```text
K-cache 自身是否有图结构？
```

它还没有回答：

```text
query token 用这个图找候选，能不能覆盖真实 full qK attention？
```

因此下一步必须进入：

```text
K-K graph candidates
vs
full qK attention top tokens / attention mass
```

只有这个 recall 通过，才能从：

```text
K-cache has graph-friendly geometry
```

升级为：

```text
K graph index can reduce qK attention computation.
```

## 1. Round2 背景

当前主问题是：

> 模型的 K-cache 是否适合建立稀疏图 index，从而让 query 不必和所有历史 K 做 full qK score？

上一轮已经得到一个偏正面的判断：

> Qwen3-0.6B 的 K-cache 具备建图的初步好性质，但还没有证明 graph candidate 能召回真实 qK attention mass。

上一轮支持这个判断的三个核心观察是：

1. **K 相似性有可稀疏化结构。**  
   Centered K-space 里 top-k neighbor similarity 有 rank decay，不是所有边都同等强。

2. **高相似 K 边不只是局部边。**  
   Centered K graph 中存在稳定比例的 long-range edges，尤其在部分 head 中更明显。

3. **K common direction 不决定 qK attention 选择性。**  
   Raw K-K cosine 很高主要来自 common direction；但这个 common direction 对同一 query 的所有 token 只贡献共同 bias，会被 softmax 抵消。Attention ranking 来自 centered / residual K。

相关文档：

```text
fdong_seq_compress/k_cache_graph_round1_findings.md
main_seq_compress/project_overview.md
```

## 1. Round2 新问题

上一轮主要用 centered cosine 作为 K-K graph edge score。Round2 进一步澄清：

> 如果最终目的是服务 qK score，那么 K-K 相似性指标是否应当更直接地使用 L2 distance？

数学依据：

```text
s1 - s2 = q · k1 - q · k2 = q · (k1 - k2)
```

因此：

```text
|q · k1 - q · k2| <= ||q|| ||k1 - k2||
```

这说明：

> K-K L2 distance 小，可以直接保证两个 K 对任意 q 的 score 不会差太远。

相比之下：

- cosine 衡量方向邻近；
- dot product 混合方向和 norm，容易产生 norm-driven hub；
- L2 distance 更直接对应 qK score stability。

不过 L2 与 centering 有一个关键性质：

```text
||(k_i - c) - (k_j - c)|| = ||k_i - k_j||
```

所以对 K-K L2 distance 而言，subtract 同一个 mean vector 不改变距离。Centering 对 cosine 很重要，但对普通 pairwise L2 本身不改变结果。

## 2. Round2 代码更新

### 2.1 K-neighbor metric 扩展

更新文件：

```text
fdong_seq_compress/src/run_k_similarity_graph_probe.py
```

现在支持：

```text
SIMILARITY=cos
SIMILARITY=dot
SIMILARITY=l2
```

语义：

```text
cos: higher is closer
dot: higher is closer, diagnostic only
l2:  lower is closer
```

注意：为了兼容旧 CSV，`summary_by_layer.csv` 里仍然使用 `similarity` 字段名；当 `SIMILARITY=l2` 时，这些值表示 L2 distance，越小越近。

### 2.2 Histogram bin 自动化

旧版本默认 histogram bins 是：

```text
-1.0:1.0:0.05
```

这只适合 cosine，不适合 dot / L2。Round2 改为：

```text
HIST_BINS=auto
```

脚本会根据当前 metric 的值自动选择 histogram bin。

### 2.3 Sweep 默认加入 L2

更新文件：

```text
fdong_seq_compress/scripts/run_k_graph_metric_sweep.sh
```

默认从：

```text
SIMILARITIES="cos dot"
```

改为：

```text
SIMILARITIES="cos dot l2"
```

因此新的 sweep 会覆盖：

```text
top_k: 10, 20, 50
similarity: cos, dot, l2
analysis_level: token, head
centered: true
max_tokens: 1000
```

### 2.4 Max position sanity check

Round2 加入了一个重要防护：

```text
seq_len <= model.config.max_position_embeddings
```

涉及文件：

```text
fdong_seq_compress/src/run_k_similarity_graph_probe.py
fdong_seq_compress/src/run_qk_common_direction_probe.py
```

如果 tokenized sequence length 超过模型配置的最大 position range，默认直接报错。

原因：

> 如果输入长度超过模型/实验代码支持的 max position，position id / RoPE / chunk reset / modulo 可能主导 K similarity，导致远距离相似边是假结构。

输出 `summary.json` 现在会记录：

```text
model_max_position_embeddings
seq_len_within_model_max_position_embeddings
```

如果刻意要测试 extrapolation，可以显式设置：

```bash
ALLOW_LONGER_THAN_MODEL_MAX=1
```

但普通几何实验不建议开启。

## 3. 已验证状态

已完成：

```bash
.venv-transformers451/bin/python -m py_compile \
  fdong_seq_compress/src/run_k_similarity_graph_probe.py \
  fdong_seq_compress/src/run_qk_common_direction_probe.py
```

已完成 L2 smoke test：

```bash
MAX_TOKENS=32 \
LAYERS=0 \
TOP_K=5 \
ANALYSIS_LEVEL=token \
SIMILARITY=l2 \
OUTPUT_DIR=fdong_seq_compress/outputs/k_l2_probe_smoke \
DEVICE=cpu \
bash fdong_seq_compress/scripts/run_k_similarity_graph_probe.sh
```

Smoke test 成功，并在 `summary.json` 中确认：

```text
model_max_position_embeddings: 40960
seq_len_within_model_max_position_embeddings: true
```

## 4. 如何运行 Round2 L2 实验

只跑 L2 sweep：

```bash
SIMILARITIES="l2" \
bash fdong_seq_compress/scripts/nohup_run_k_graph_metric_sweep.sh
```

跑完整 metric sweep：

```bash
bash fdong_seq_compress/scripts/nohup_run_k_graph_metric_sweep.sh
```

监控最新 sweep：

```bash
cat fdong_seq_compress/outputs/k_graph_metric_sweep_latest_path.txt
```

```bash
tail -f "$(cat fdong_seq_compress/outputs/k_graph_metric_sweep_latest_path.txt)/nohup.log"
```

看完成的实验：

```bash
cat "$(cat fdong_seq_compress/outputs/k_graph_metric_sweep_latest_path.txt)/manifest.csv"
```

## 5. Round2 结果解释框架

读 Round2 结果时，不应只问“L2 最近邻是否存在”，而应围绕最终目标：

> L2 K graph 是否更适合服务 qK score indexing？

本轮已经按以下角度完成了第一批判断：

### 5.1 Rank-k L2 decay

看：

```text
rank10_l2
rank20_l2
rank50_l2
```

如果 rank-k L2 随 rank 增长明显变大，说明每个 node 的有效邻域有限，适合 sparse graph。

如果 rank10 到 rank50 距离差别不大，说明 bucket 边界弱。

### 5.2 L2 neighbor 的 sequence distance

继续看：

```text
distance_frac_ge_128
distance_frac_ge_256
distance_p50
distance_p95
```

本轮观察到：L2 nearest neighbors 也有 long-range edges，说明非局部 graph 不是 cosine artifact。

如果未来新模型/新数据上 L2 nearest neighbors 几乎全是局部边，则说明 cosine graph 的部分 long-range 结构可能主要是方向相似，而不是 score-stability 相似。

### 5.3 L2 graph 的 in-degree / hubness

看：

```text
indegree_top1pct_edge_frac
indegree_top5pct_edge_frac
indegree_max
indegree_zero_frac
```

理想信号：

```text
moderate long-tail hubness
not dominated by a few norm artifacts
```

Dot graph 中曾出现极端 hubness，因此 L2 graph 可以帮助判断哪些 hub 更可靠。

### 5.4 Cosine vs L2 的一致性

核心比较：

```text
centered cosine graph
vs
L2 graph
```

本轮两者都显示：

```text
rank decay
long-range edges
reasonable hubness
```

因此 K-cache graph evidence 比 Round1 更强。

如果 cosine 支持而 L2 不支持，则要谨慎：cosine 的方向邻近未必足以保证 qK score stability。

## 6. Round2 完成后的研究判断

Round2 完成后的判断是：

> centered cosine graph 和 L2 graph 都支持 K-cache 具有 graph-friendly geometry。

这比 Round1 更强，因为 Round1 主要依赖 centered cosine；Round2 进一步说明：

```text
K-cache 的局部边、长程边、hub/anchor、layer/head 差异
不是某一个文本域、某一个 top-k、某一个 metric 下的偶然现象。
```

但是这个判断仍然停留在 K-K geometry 层面。真正的系统目标是：

```text
query q_t 到来时，
能否通过已有 K graph 找到 full qK attention 真正重要的历史 token？
```

因此现在的研究状态应表述为：

```text
K-cache graph indexing: geometrically motivated, not yet attention-validated.
```

这意味着下一步必须做两个层次的验证。

### 6.1 推理时 graph candidate 是否真的有效

先不训练新模型，直接构造 inference-time candidate generator：

```text
输入：已有 K-cache
构图：centered cosine / L2 top-k graph
查询：当前 q_t
候选：local window + graph candidates
验证：和 full qK attention 比 recall
```

建议先做最小版本：

```text
1. 对历史 K 建 causal top-k graph
2. 对当前 q_t 先看 recent local window
3. 从 local window 里的高分 K 节点扩展 1-hop / 2-hop graph neighbors
4. 只在候选集合上算 qK
5. 和 full qK attention 对比
```

关键指标：

```text
attention mass recall
top attention token recall
candidate size / compression ratio
CE delta after masking non-candidates
per-layer/head recall
```

如果这个实验成功，说明：

```text
现有 pretrained model 的 K-cache 已经自然形成可用图索引。
```

如果失败，也不等于图方向错，而是说明：

```text
K-K 自相似结构存在，但它未必和 query-time attention routing 对齐。
```

这时要进入训练时约束。

### 6.2 单句 / 单文档可视化

为了理解图到底长什么样，应当拿一条具体长文本做可视化，而不是只看 CSV。

可视化对象：

```text
选一个 domain text
选 2-3 个代表性 layer/head
画 token node + K-neighbor edge
```

建议至少画三类 head：

```text
local head: 例如 L27/H0
long-range head: 例如 L6/H3 或 L15/H1
mixed head: median distance 居中的 head
```

要看的不是一张“漂亮图”，而是：

```text
局部 head 是否像链 / banded graph
长程 head 是否连接章节、重复实体、结构位置、代码符号
hub token 是否对应分隔符、标题、实体、段落中心或特殊 token
1-hop / 2-hop 扩展是否迅速覆盖语义相关区域
```

如果可视化显示 long-range edges 有文本语义或结构语义，这会增强 graph index 的可信度。

如果可视化显示 long-range edges 主要连到无意义 token / punctuation / template artifact，就要重新审视 graph metric。

### 6.3 如果推理时 graph 失败，进入训练时约束

如果 inference-time graph candidate recall 不够，可能原因是：

```text
pretrained K-space 有图结构，
但训练时没有要求这个图结构服务 sparse retrieval。
```

这时合理方向不是继续手调 graph，而是考虑训练时让模型学出“训推一致”的 K-space：

```text
训练时加入 graph-aware attention mask
训练时加入 K clustering / routing auxiliary loss
训练时加入 query-to-block / query-to-anchor recall objective
训练一个 CSA-like lightning indexer
训练 K-side index + V-side faithful payload 的结构
```

这和当前 framework 的大方向一致：

```text
K 负责 index / routing
V 保持 high-fidelity content
```

也就是说，Round2 的几何结论可以支持两条路线：

```text
路线 A：pretrained K-space 已经够好，做 inference-time graph index
路线 B：pretrained K-space 只提供启发，需要训练时显式塑造 graph-indexable K-space
```

## 7. Important Caveats

实验：考虑所有可能的维度，做地毯式搜索实验，看“k cache 是否适合构建图索引”这个结论，在各维度上是否 consistent？
1、seq len：例如，序列增加到 100w；
2、common：去/不去除 mean bias。（我们期待添加上 mean bias 之后，k 难以构建图索引了）
3、layer；
4、head；
5、similarity metric：度量相似性的指标，我觉得还是应当用 k,k 之间的 l2 norm，因为我把第 n 个 token 的 k 添加到前面所有 k 的图 index 里面，是为了服务第 n+1 个 token 能快速索引 k，所以我无法用第 n+1 个 token 的 q 去度量前面的 k 之间的相似性；

期待的现象：
1、k 相似性是否稀疏：和第 n 个 k 高相似的 token 有多少？我们期待某种“稀疏聚类”结构，于是后续每个 q 可以只和前面少数的 k/v 做 attention 计算；
2、相似 k 的距离：期待和当前 k 有关系的前面的 k 应当离我远近都有，这样 deepseek 的那种 local 压缩方式就不是最好的，我们应当能做的更好；

对 Solution 的启发：
1、构建 k 聚类，先和 k center 做索引，再和每个 cluster 内的 k 做 attention（现在有人做过，我们的区别是我们去掉了 mean bias?）；
2、层次化 (hierarchical) 的 k index；（目前不急，我们先做一层，一层做好后，多层不过是技术上的优化）
3、借鉴系统领域对 kv cache tree indexing 的工作。

## 8. 仍可补充但不再是第一优先级的 Sweep

Round2 已经完成 L2 metric、seq-len scaling、layer/head selection、domain sweep。下面这些 sweep 仍有价值，但现在优先级低于 query-attention recall 和可视化。

### 8.1 Seq-len Scaling Sweep

目的：

> 验证 graph-friendly geometry 是否随 sequence length 增长保持稳定，而不是 1k token 下的局部现象。

已完成默认配置：

```text
MAX_TOKENS = 1000, 2000, 4000, 8000
SIMILARITY = cos, l2
ANALYSIS_LEVEL = head
TOP_K = 10
CENTER_TOKENS = 1
```

未来如果要进一步补充，可以加：

```text
MAX_TOKENS = 12000
TOP_K = 20
sampled longer contexts
```

看：

```text
rank-k decay 是否随长度稳定
long-range edge fraction 是否增加或消失
hubness 是否变得极端
graph-friendly heads 是否一致
```

注意：更长序列的 exact K-K pairwise 复杂度是 `O(N^2)`，不能直接跳到 `100w`。真实长序列应走：

```text
1k -> 4k -> 12k -> sampled 100k / block-level approximate
```

### 8.2 Debiasing / Common-direction Sweep

目的：

> 验证图结构是否来自 attention-relevant residual K，而不是 common direction artifact。

建议比较：

```text
raw K
centered K
remove top-1 PC
remove top-4 PCs
remove top-8 PCs
whitened K
```

期待现象：

```text
raw K: graph 过密，区分度弱
centered K: rank decay 更清晰
PC-removed / whitened K: 如果仍有 rank decay + long-range edges，graph 证据更强
```

当前代码已经支持：

```text
KEY_TRANSFORM=raw
KEY_TRANSFORM=center
KEY_TRANSFORM=remove_pc PC_REMOVE_COUNT=1/4/8
KEY_TRANSFORM=whiten
```

可直接运行：

```bash
bash fdong_seq_compress/scripts/run_k_transform_sweep.sh
```

### 8.3 Layer / Head Selection Sweep

目的：

> 找到稳定 graph-friendly 的层和头，而不是默认所有 layer/head 都适合建图。

已完成默认配置：

```text
ANALYSIS_LEVEL = head
LAYERS = all
HEADS = all
SIMILARITY = cos, l2
TOP_K = 10, 20, 50
```

输出后按以下指标筛选 heads：

```text
rank decay strong
distance_frac_ge_128 / ge_256 high
hubness moderate
zero in-degree not too high
not dot/norm-artifact dominated
```

后续 qK recall 先在这些 selected heads 上做，不要一开始平均所有 heads。

### 8.4 Graph Construction Sweep

目的：

> 不是所有 KNN graph 都适合 query routing，需要比较不同图构造方法。

候选：

```text
directed causal top-k graph
mutual KNN graph
radius-threshold graph
high in-degree anchor graph
medoid / k-medoids cluster graph
k-means center graph
local-window + graph-long-edge hybrid
```

看：

```text
edge count / candidate budget
long-range coverage
hub concentration
anchor neighborhood size
graph connectivity / component size
```

这一步之后才能定义 query-time candidate generation。

当前代码已支持两类轻量 graph construction 对比：

```text
GRAPH_MODE=topk
GRAPH_MODE=radius
```

并且每次都会输出：

```text
graph_structure_summary_by_layer.csv
```

包括：

```text
edge count / avg outdegree
largest component fraction
component count
isolated node fraction
local <=8 edge fraction
long >=128 / >=256 edge fraction
```

可直接运行：

```bash
bash fdong_seq_compress/scripts/run_k_graph_construction_sweep.sh
```

更复杂的 `medoid / k-means / hierarchical tree` 暂时不实现，因为它们已经接近 solution design。当前阶段先用 top-k / radius 图理解几何结构即可。

### 8.5 Baseline Sweep

目的：

> 防止把普通高维最近邻极值、局部平滑或文本重复误判成有用图结构。

需要至少比较：

```text
local window baseline
random same-count candidate baseline
row-permuted K baseline
norm-preserving random direction baseline
Gaussian same-dim baseline
block-summary baseline
```

判断标准：

```text
K graph 必须显著优于 local-only 和 random same-count，才值得进入系统设计。
```

### 8.6 Data / Domain Sweep

目的：

> 检查 K graph 结构是不是 synthetic text artifact。

本轮已经覆盖：

```text
synthetic long technical report
codebase / query engine style text
textbook chapter
news / supply-chain dossier
dialogue + tool transcript
```

看：

```text
graph-friendly heads 是否跨数据稳定
long-range edges 是否在代码/多文档中更强
hub token 是否对应结构性文本位置
```

本轮主要结论是：

```text
局部边 + 长程边 + layer/head 差异在 5 类文本上都存在。
```

### 8.7 Query-attention Recall Sweep

这是 L2 和 geometry sweep 后最关键的一步。

目的：

> 验证 K graph candidates 是否真的覆盖 full qK attention。

候选生成方式：

```text
anchor-only
anchor + 1-hop neighborhood
top graph clusters
local recent window + graph candidates
centered cosine candidates
L2 candidates
cos ∩ L2 candidates
cos ∪ L2 candidates
```

指标：

```text
attention mass recall
top attention token recall
candidate size / compression ratio
per-layer/head recall
CE delta if masking non-candidates
```

只有这个 sweep 通过，才能从：

```text
K-cache has graph-friendly geometry
```

升级到：

```text
K graph index can reduce qK attention computation.
```

## 9. Round2 后推荐执行顺序

Round2 已经完成：

```text
1. L2 metric sweep
2. seq-len scaling sweep
3. layer/head selection sweep
4. data/domain sweep
```

因此下一阶段不要继续只做更多几何表格。推荐顺序改为：

```text
1. Query-attention recall sweep
2. 单条长文本的 K graph 可视化
3. Minimal inference-time graph candidate generator
4. 如果 inference-time recall 不够，再设计 training-time constraint
5. 最后才进入更复杂的 graph construction / system optimization
```

### 9.1 第一优先级：query-attention recall

这是最关键 gate。

要回答：

```text
K-K graph candidates 是否覆盖 full qK attention 真正看的 token？
```

候选方式从简单到复杂：

```text
local window only
K-K top-k graph 1-hop
K-K top-k graph 2-hop
local window + graph 1-hop
centered cosine candidates
L2 candidates
cos ∩ L2 candidates
cos ∪ L2 candidates
selected-head-only candidates
```

如果这些候选能用较小 candidate size 覆盖大部分 attention mass，那么图索引方向进入 inference-time prototype。

### 9.2 第二优先级：单文档 graph visualization

做一条具体长文本的图可视化，目的是理解结构，不是为了展示。

建议选：

```text
long_textbook_distributed_systems
或
long_codebase_query_engine
```

因为这两类文本更容易解释 long-range edge：

```text
概念重复
章节结构
代码符号
API 名称
错误处理路径
```

建议画：

```text
L27/H0: local head
L6/H3: long-range head
L15/H1: long-range head
一个 mixed head
```

可视化形式可以从简单开始：

```text
token position as x-axis
edge as arc
edge color by distance / metric
node size by in-degree
hover text showing token string
```

进一步可以画 tree-like expansion：

```text
选一个 query-adjacent seed token
画 1-hop / 2-hop K graph expansion
看它像局部链、星型 hub、还是跨段落树
```

### 9.3 第三优先级：minimal inference-time graph index

如果 recall sweep 有正信号，就实现一个最小推理时方案：

```text
每个 layer/head 维护 K graph
新 token 到来时，把它插入 graph
query 先从 local window 或 anchor nodes 找入口
扩展 graph neighbors 得到 candidates
只对 candidates 做 qK attention
```

先不要做复杂系统优化。先回答：

```text
候选集合大小下降多少？
attention recall 下降多少？
loss / perplexity / CE delta 下降多少？
哪些 layer/head 最适合 sparse？
```

### 9.4 如果推理时失败，转向训练时结构

如果 inference-time graph index 失败，可能说明：

```text
自然 pretrained K-space 有图结构，
但这个结构不是为 sparse retrieval 训练出来的。
```

这时下一步是训练时让模型形成可索引 K-space：

```text
训练时随机 mask 非 graph candidates
训练时让 q 更依赖 graph-retrieved K/V
训练 auxiliary indexer 预测 useful K blocks
训练 K clustering / block routing loss
借鉴 CSA / MCA：K-side learned index，V-side high-fidelity content
```

这条路线的判断标准是：

```text
训练后 K graph recall 是否显著提升
训练后 sparse candidate attention 的 CE delta 是否显著降低
训推是否一致
```

### 9.5 暂缓事项

以下事情不是当前第一优先级：

```text
remove PC / whitening sweep
复杂 graph construction sweep
100k / 1M token 级别 scaling
系统级 cache eviction / online ANN optimization
```

原因不是它们不重要，而是现在还没有回答最核心问题：

> K graph candidates 能不能服务真实 query attention？

如果这个问题答案是正面的，再做复杂系统优化才有意义。
