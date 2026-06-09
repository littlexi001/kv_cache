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

### 7.6 layer_shared 版本实验结果与瓶颈分析

在上一版 `attention_head` 粒度中，tree retrieval 仍然需要对每层、每个 KV head
分别生成候选集合。profile 显示候选生成本身是主要瓶颈。为降低这部分开销，当前
实现新增了更激进的候选共享策略：

```text
candidate_granularity = layer_shared
tree_attention_impl = shared_matmul
tree_branch_counts = 5,2,2
tree_prefill = false
```

`layer_shared` 的含义是：每一层只生成一套候选 token id，并让该层所有
attention heads 共享这套候选集合。具体做法是用当前 chunk 最后一个 token
对应的所有 attention head query 均值作为 representative query，同时用所有
KV head 的 key 均值作为检索 key 表示。这样可以把候选生成调用次数从
`layers * kv_heads * eval_chunks` 降到 `layers * eval_chunks`，代价是候选集合
不再 head-specific，PPL 可能变差。

本次实验结果如下：

PPL 结果：

| mode | loss | PPL | token_count | candidate_granularity | tree_attention_impl |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 4.588919 | 98.3880 | 5000 | layer_shared | shared_matmul |
| tree | 4.720535 | 112.2283 | 5000 | layer_shared | shared_matmul |

速度结果：

| mode | prefill_seconds | eval_seconds | total_seconds | tokens_per_second | avg_candidate_tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 119.976 | 11.831 | 131.808 | 422.60 | - |
| tree | 119.976 | 7.843 | 127.820 | 637.48 | 3035.93 |

从 eval 阶段看，tree 版本从 `11.831s` 降到 `7.843s`：

```text
eval speedup = 11.831 / 7.843 = 1.51x
```

但总时间只从 `131.808s` 降到 `127.820s`，原因是当前评测已经共享 prefill，
两条分支的 `prefill_seconds` 都是 `119.976s`。因此本实验主要衡量 eval
阶段的注意力替换收益，而不是端到端 prefill 加速。

质量上，tree 版本相比 baseline：

```text
delta loss = 4.720535 - 4.588919 = +0.1316
PPL ratio = 112.2283 / 98.3880 = 1.141
```

也就是 PPL 增加约 `14.1%`。这说明 `layer_shared` 的速度收益是以更粗粒度
候选集合为代价换来的。相比 per-head 检索，它减少了候选生成开销，但也损失了
不同 head 的 query-dependent 选择能力。

本次平均候选 token 数为：

```text
avg_candidate_tokens = 3035.93
```

如果上下文约为 200k token，这仍然约等于只使用 `1.5%` 的历史 token。注意，
这个比例并不直接等价于理论加速比，因为当前实现仍然有候选生成、gather、
mask、softmax 和 PyTorch kernel 调度等额外开销。

profile 结果显示，当前主要瓶颈为：

| stage | seconds | calls | seconds_per_call | percent_profiled |
| --- | ---: | ---: | ---: | ---: |
| shared_matmul/chunk_candidate_ids | 3.745 | 1120 | 0.003344 | 57.50% |
| shared_matmul/mid_topk_fast | 0.510 | 1120 | 0.000455 | 7.82% |
| shared_matmul/leaf_topk_fast | 0.385 | 1120 | 0.000344 | 5.92% |
| shared_matmul/leaf_tokens_fast | 0.357 | 1120 | 0.000319 | 5.48% |
| shared_matmul/select_kv | 0.252 | 1120 | 0.000225 | 3.86% |
| shared_matmul/mask | 0.203 | 1120 | 0.000181 | 3.11% |
| shared_matmul/qk_matmul | 0.093 | 1120 | 0.000083 | 1.43% |
| shared_matmul/softmax | 0.160 | 1120 | 0.000143 | 2.45% |
| shared_matmul/av_matmul | 0.091 | 1120 | 0.000081 | 1.40% |

这组 profile 的关键结论是：当前慢的不是 `QK` 或 `AV` 矩阵乘法。`qk_matmul`
和 `av_matmul` 合计只有约 `0.18s`，真正的大头仍然是
`chunk_candidate_ids`，即树检索候选生成。

期间也尝试过 `triton_shared`，即用 Triton 将候选集合上的
`QK -> softmax -> AV` 融合成一个 kernel。但实验显示该方向更慢：

```text
triton_shared/chunk_candidate_ids = 13.012s
triton_shared/fused_attention = 28.705s
```

原因是该 kernel 仍然需要根据随机 candidate id 访问 K/V，访存模式不如
PyTorch `shared_matmul` 路径中的 batched matmul 友好。因此当前结论是：
不应继续优先优化 attention matmul，而应优先优化候选生成。

当前实现还去掉了 `shared_matmul` 中的 `repeat_kv`。原先为了 GQA 会执行：

```text
key_states.repeat_interleave(group_size, dim=1)
value_states.repeat_interleave(group_size, dim=1)
```

这会把 KV heads 显式复制到 attention-head 维度，带来额外内存开销。现在改为
直接用：

```text
kv_head_index = attention_head // group_size
selected_keys = key_states[batch_index, kv_head_index, candidate_ids]
selected_values = value_states[batch_index, kv_head_index, candidate_ids]
```

随机张量对比验证显示新旧 selected K/V 完全一致，因此这是等价优化。

当前阶段的结论如下：

1. `layer_shared` 明显降低了候选生成次数，使 tree eval 相比 baseline eval
   达到约 `1.51x` 加速。
2. 该加速仍远低于“只使用 1.5% token”对应的理论上限，因为候选生成和 gather
   仍然是主要开销。
3. `layer_shared` 带来约 `14.1%` PPL 增加，说明所有 head 共享候选集合过于粗糙，
   但它是一个有用的速度上界实验。
4. 下一步如果继续追求大幅速度提升，应优先把 `chunk_candidate_ids` 继续压低，
   例如减少树层级 top-k 次数、复用 layer-level 候选、改成更粗的 block-level
   检索，或者将候选生成本身写成 CUDA/Triton kernel。
