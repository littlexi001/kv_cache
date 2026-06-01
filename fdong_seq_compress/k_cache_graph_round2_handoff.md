# K-cache Graph Index Round2 Handoff

Date: 2026-06-01

## 0. Round2 背景

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
fdong_seq_compress/qwen3_k_graph_metric_sweep_findings.md
fdong_seq_compress/qwen3_qk_common_direction_findings.md
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

## 5. Round2 分析时应回答的问题

下一轮读 L2 实验结果时，不应只问“L2 最近邻是否存在”，而应围绕最终目标：

> L2 K graph 是否更适合服务 qK score indexing？

建议重点看：

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

如果 L2 nearest neighbors 也有 long-range edges，说明非局部 graph 不是 cosine artifact。

如果 L2 nearest neighbors 几乎全是局部边，则说明 cosine graph 的部分 long-range 结构可能主要是方向相似，而不是 score-stability 相似。

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

如果两者都显示：

```text
rank decay
long-range edges
reasonable hubness
```

则 K-cache graph evidence 更强。

如果 cosine 支持而 L2 不支持，则要谨慎：cosine 的方向邻近未必足以保证 qK score stability。

## 6. 当前研究判断

Round2 更新后的当前判断是：

> centered cosine graph 已经显示 K-cache 具有图结构潜力；但为了服务 qK score indexing，必须进一步验证 L2 graph 是否也具有 rank decay、long-range edges 和 hub/anchor structure。

如果 L2 结果也支持：

```text
K-cache has graph-friendly geometry under score-stability metric.
```

那么下一步应进入真正的 attention recall 实验：

```text
K graph candidates
-> compare against full qK attention
-> measure attention mass recall / top-token recall / candidate budget
```

这是从“几何上适合建图”走向“真的能降低 qK 计算成本”的必要一步。

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