# Section 16. KV Tree Retrieval PPL 实验结果

本节记录 Section 15 中层次化 tree retrieval 策略进入 Phase B 之后的
直接 loss/PPL 实验结果。

## 1. 通用实验配置

下面三组实验使用相同的名义 tree retrieval 预算：

```text
model: Qwen3-0.6B
layers: 全部 28 层
kv_heads: 全部 8 个 KV head
boundary_fraction: 0.005
leaf_fraction: 0.001
leaf_size: 0
tree_fanout: 10
tree_branch_counts: 5,2,2
candidate_granularity: attention_head
```

候选集合比例大约为：

```text
prefix 0.5% + recent 0.5% + 5 * 2 * 2 个 leaf * 0.1%
= 每个 attention head 约 3% 可见 token
```

其中 `prefix` 和 `recent` 是强制保留的边界 token。tree 只在去掉
prefix/recent 后的 middle 区域里检索。

![KV Tree Retrieval 结构图](assets/section16/kv_tree_retrieval_structure.png)

图 16-1：KV tree retrieval 的候选集合构造流程。每层、每个 KV head 在
middle 区域上构建连续 block 的 K-cache 层次树；每个 query 使用
`q_t · center(node)` 自顶向下选择 top branch，被选中 leaf 内的 token 与
强制保留的 prefix/recent token 合并为候选集合 `S_t`，再用于 sparse
attention 或 PPL 评估。

## 2. 实验结果

| Run | token_count | tree_attention_impl | tree_prefill | baseline loss | tree loss | delta loss | baseline PPL | tree PPL | PPL ratio |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| A | 50000/50000 | 未记录 | 未记录 | 3.606601 | 3.519489 | -0.087112 | 36.8406 | 33.7672 | 0.9166 |
| B | 50000/5000 | sparse_gather | 未记录 | 3.814184 | 3.891407 | +0.077223 | 45.3397 | 48.9798 | 1.0803 |
| C | 200000/1000 | sparse_gather | false | 8.664809 | 8.749083 | +0.084275 | 5795.3366 | 6304.9070 | 1.0879 |

## 3. 结果解读

Run A 是目前最强的正向结果。在约 3% token budget 下，tree 版本的
loss 相比 baseline 降低 `0.0871`，PPL 下降约 `8.34%`。由于它评估了
50k 个 token，这一组在当前结果中统计稳定性最好。

Run B 和 Run C 在同样的名义 tree 预算下出现了轻微退化。loss 增加约
`0.077-0.084`，对应 PPL 大约增加 `8%`。这两组评估 token 数更少，
尤其 Run C 只有 1000 个 eval token，因此更容易受到文本片段、prefill
长度和 eval 边界位置的影响。

当前最重要的质量结论是：即使在全部层、全部 KV head 上都使用 tree
retrieval，并且每个 attention head 只保留约 3% 可见 token，这个策略也
没有出现灾难性 PPL 崩坏。根据文本片段和运行模式不同，它的表现范围大致是：

```text
最好：PPL 比 baseline 下降约 8%
较差：PPL 比 baseline 上升约 8-9%
```

这说明 tree retrieval 作为候选集合策略是有继续研究价值的，但还需要更
严格的同文本、同 token 数、同 prefill 设置下的对照实验。

## 4. 运行速度记录

50k-token tree run 的日志中记录了：

```text
ppl tree chunk 195/196: tokens 99664-99919
ppl tree chunk 196/196: tokens 99920-99999
timer tree_eval: 1119.632s
timer tree_total: 1568.806s
eval throughput: 44.66 tokens/s
```

这说明当前实现仍然应该被视为 retrieval policy 的质量验证实验，而不是
生产级 sparse attention 加速实现。即使已经加入 `sparse_gather`，当前版本
仍然存在 tree 构建、候选选择、gather 以及 PyTorch eager kernel 调度开销。

## 5. 可比性注意事项

这三条记录不能直接解释成单一的单调趋势，因为它们的 eval token 数不同，
并且至少 Run C 使用了 `tree_prefill=false`。

后续需要做更严格的 controlled comparison：

1. 固定同一文本片段和同一 token 数，对比 `tree_attention_impl=mask` 与
   `tree_attention_impl=sparse_gather`。
2. 固定同一文本片段和同一 token 数，对比 `tree_prefill=true` 与
   `tree_prefill=false`。
3. 在另一个文本片段上重复 50k-token run，确认 Run A 的 PPL 改善是否稳定。
4. 每次实验同时记录 candidate fraction、PPL 和运行时间。

## 6. 下一步 sweep 建议

当前配置约为 3% token budget。建议下一步围绕 3-6% 做小范围 sweep：

```text
boundary_fraction=0.005, tree_branch_counts=5,2,2  # 约 3%
boundary_fraction=0.010, tree_branch_counts=5,2,2  # 约 4%
boundary_fraction=0.005, tree_branch_counts=5,3,2  # 约 4%
boundary_fraction=0.005, tree_branch_counts=5,3,3  # 约 5.5%
```

如果 Run A 中 50k-token 的 PPL 改善能在不同文本片段上复现，那么当前 tree
retrieval policy 就值得继续推进到更优化的 sparse attention 实现。

## 7. 当前实现版本：shared-matmul tree PPL

后续代码已经从最初的 `mask` / `sparse_gather` 原型，演进到一个更偏工程验证的
`shared_matmul` 版本。当前实验目标是：在超长 context 下，用 K-cache tree
retrieval 选出少量候选 KV token，替代 full attention，并同时比较 PPL 退化、
eval 速度和平均候选 token 数。

对应脚本：

```text
ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_ppl_only.sh
```

当前默认配置已经改为：

```text
prefill_tokens = 20000
eval_tokens = 5000
chunk_size = 128
tree_prefill = false
tree_attention_impl = shared_matmul
tree_branch_counts = 5,2,2
leaf_fraction = 0.001
candidate_granularity = attention_head
profile_tree_stages = true
```

如果运行 200k context，需要通过环境变量覆盖：

```bash
PREFILL_TOKENS=200000 \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_ppl_only.sh
```

### 7.1 Boundary 规则

虽然脚本里仍保留 `--boundary_fraction` 参数用于 metadata 兼容，但当前 PPL
tree 路径实际使用固定 token 数规则：

```text
key_count <= 10000: prefix 50 + recent 50
key_count > 10000:  prefix 500 + recent 500
```

因此在 200k context 下，固定保留前 500 个 token 和最近 500 个 token。
tree 只在去掉 prefix/recent 后的 middle 区域中检索。

### 7.2 shared-matmul 检索与计算方式

当前 `shared_matmul` 不是每个 query 单独检索，而是按 chunk 共享候选集合：

```text
每个 layer
每个 KV head
每个 eval chunk
使用 chunk 最后一个 query 作为 representative query
在 middle 区域做 tree retrieval
得到该 chunk/head 共用的一组 candidate tokens
chunk 内所有 query 共用这组候选
```

由于后续仍然施加 causal mask，chunk 中较早的 query 不会看到未来 token。

tree 的 leaf 是连续 token block：

```text
leaf_size = ceil(leaf_fraction * key_count)
```

例如 200k context、`leaf_fraction=0.001` 时：

```text
leaf_size ≈ 200 tokens
```

`tree_branch_counts=5,2,1` 时选出：

```text
5 * 2 * 1 = 10 leaves
tree middle tokens ≈ 10 * 200 = 2000
prefix/recent = 500 + 500
total candidates ≈ 3000
```

这与最新实验中的 `avg_candidate_tokens≈2990` 基本一致。

attention 计算方式为：

```text
candidate_ids: [batch, heads, shared_candidates]
K_selected:    [batch, heads, shared_candidates, head_dim]
Q_chunk:       [batch, heads, query_count, head_dim]
scores = Q_chunk @ K_selected^T
softmax
output = attention @ V_selected
```

这比 per-query `sparse_gather` 更接近 GPU 高效的 batched matmul 路径。

### 7.3 已完成的工程优化

当前实现相对早期版本已经做过以下优化：

1. 去掉 dense `keep` mask，不再先构造 `[heads, query, key]` mask 再反向
   `topk` 得到候选 token。
2. 去掉 sort 去重，依赖 prefix/recent 与 middle tree 区域不重叠来避免
   大部分重复。
3. 增加 `avg_candidate_tokens` 统计，写入 `timing_by_mode.csv`。
4. 增加 `profile_tree_stages` 分段计时，写入 `tree_stage_profile.csv`。
5. 新增 `shared_matmul` 路径，chunk 内 query 共用一组候选 token。
6. 将 `shared_matmul` 从“每个 query 检索后做 union”改成“每个 chunk 只用
   最后一个 query 检索一次”。
7. 去掉 `shared_matmul` 中的 `prefix_sum/cumsum` 和 range tensor 构造，
   改成直接在 middle 区域按连续 leaf block 计算 leaf/mid/big centers。

### 7.4 最新 200k / 5k 结果

最新一组 200k prefill、5k eval 的结果如下：

```text
prefill_tokens = 200000
eval_tokens = 5000
tree_branch_counts = 5,2,1
tree_attention_impl = shared_matmul
tree_prefill = false
```

PPL 结果：

| mode | loss | PPL | token_count | avg_candidate_tokens |
| --- | ---: | ---: | ---: | ---: |
| baseline | 8.458820 | 4716.4910 | 5000 | full attention |
| tree | 8.543097 | 5131.2134 | 5000 | 2990.42 |

速度结果：

| mode | prefill_seconds | eval_seconds | total_seconds | tokens_per_second |
| --- | ---: | ---: | ---: | ---: |
| baseline | 467.392 | 23.846 | 491.239 | 209.68 |
| tree | 462.647 | 22.281 | 484.927 | 224.41 |

这一组说明，在 200k context 下，tree 版本平均只使用约 2990 个候选 token：

```text
2990 / 200000 ≈ 1.5%
```

质量上，tree 相比 baseline：

```text
delta loss ≈ +0.0843
PPL ratio ≈ 1.088
```

也就是 PPL 增加约 8.8%。速度上，tree eval 比 baseline eval 略快：

```text
23.846 / 22.281 ≈ 1.07x
```

这说明 tree retrieval 在超长 context 下开始出现实际速度收益，但当前实现仍然
不是生产级 fused sparse attention kernel，因此 1.5% 候选 token 并不会直接
转化为 60x 级别加速。

### 7.5 当前运行建议

如果只是调 tree 参数，不需要每次重复跑 baseline。可以固定已有 baseline，
只运行 tree：

```bash
COMPUTE_BASELINE_PPL=false \
COMPUTE_TREE_PPL=true \
TREE_PREFILL=false \
PROFILE_TREE_STAGES=false \
PREFILL_TOKENS=200000 \
EVAL_TOKENS=5000 \
TREE_BRANCH_COUNTS=5,2,1 \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_ppl_only.sh
```

如果需要分析瓶颈，可以短跑打开 profile：

```bash
COMPUTE_BASELINE_PPL=false \
PROFILE_TREE_STAGES=true \
PREFILL_TOKENS=200000 \
EVAL_TOKENS=256 \
TREE_BRANCH_COUNTS=5,2,1 \
bash ymluo/projects/qwen3_kv_tree_retrieval_energy_analysis/scripts/run_ppl_only.sh
```

主要输出文件：

```text
ppl_by_tree.csv
timing_by_mode.csv
tree_stage_profile.csv
```

其中 `timing_by_mode.csv` 中的 `avg_candidate_tokens` 只对 tree 行有意义；
baseline 行的 `tokens_per_second` 不是候选 token 数。
