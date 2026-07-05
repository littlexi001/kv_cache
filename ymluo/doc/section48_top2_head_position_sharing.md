# 第 48 节：Top2 Position 在 Head 之间是否可以共享

日期：2026-07-03

## 0. 目标

前面实验显示：

```text
每个 head 只保留真实 full-QK top 2% historical tokens，质量可以接近 full attention。
```

新的问题是 top2 position selection 本身代价仍然高。如果多个 head 的 top2 position set 高度重合，就可以让其中一个 head 做 selector，其他 head 复用同一组 historical positions。

本节实验回答两个问题：

```text
1. 28*16 个 head 中，哪些 head 可以共享 top2 position selection？
2. 这种共享关系是否稳定，还是强烈依赖当前 query token？
```

## 1. 新增脚本

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_head_position_sharing.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_head_position_sharing_server.sh
```

默认服务器运行：

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_war_4k_v1 \
bash scripts/run_top2_head_position_sharing_server.sh
```

默认设置：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = data/war_and_peace_pg2600.txt
prefill_tokens = 4096
eval_tokens = 64
layers = all 28 layers
heads = all 16 heads
top_fraction = 0.02
attention = eager
dtype = float16
```

Remote-only 版本：

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_remote_war_4k_s64_r512_v1 \
REMOTE_ONLY=true \
EXCLUDE_SINK_TOKENS=64 \
EXCLUDE_RECENT_TOKENS=512 \
bash scripts/run_top2_head_position_sharing_server.sh
```

## 2. 实验定义

对每个 sampled query token、每一层、每个 head，先用真实 full-QK score 选出：

```text
S(l, h, q) = top ceil(0.02 * history_count) historical positions
```

然后把一个 head 当成 donor，另一个 head 当成 target：

```text
recall(d -> t) = |S(l,d,q) ∩ S(l,t,q)| / |S(l,t,q)|
```

这个 recall 表示：

```text
如果 target head 不再自己做 top2 selector，而是直接复用 donor head 的 positions，
target 原本真实 top2 positions 有多少被保留下来。
```

还统计 attention mass recall：

```text
mass_recall(d -> t) =
  target head 在 donor positions 上的 full attention mass
  /
  target head 在自己真实 top2 positions 上的 full attention mass
```

这比纯 set recall 更接近质量风险：如果没覆盖的 token attention weight 很小，mass recall 仍然可能高。

## 3. 输出文件

核心输出：

```text
head_sharing_donor_target.csv
```

每行是同一 layer 内一个有向 head pair：

```text
layer, donor_head, target_head
top2_recall_mean
top2_recall_std
top2_jaccard_mean
attention_mass_recall_mean
top2_recall_ge_0p70_fraction
top2_recall_ge_0p80_fraction
top2_recall_ge_0p90_fraction
```

`head_best_donors.csv`：

```text
对每个 target head，找最适合复用的 donor head。
```

`shared_head_groups.csv`：

```text
按 recall threshold 做 greedy star grouping。
每个 group 只需要 representative_head 运行 top2 selector，
member_heads 复用 representative_head 的 positions。
```

`sharing_threshold_summary.csv`：

```text
不同共享阈值下：
selectors_without_sharing
selectors_with_sharing
selectors_saved
selector_reduction_fraction
shared_target_top2_recall_mean
shared_target_attention_mass_recall_mean
shared_target_stable_fraction_mean
```

`query_group_recall_by_threshold.csv`：

```text
固定 aggregate statistics 得到的共享分组后，
逐 query token 统计共享 recall 和 mass recall。
```

这个文件用来判断共享策略是否和 query token 相关。如果某些 query token 上 recall 明显掉下去，说明共享关系不是静态稳定的。

## 4. 读结果时重点看什么

第一步看整体可共享性：

```text
sharing_threshold_summary.csv
```

重点字段：

```text
selector_reduction_fraction
shared_target_top2_recall_mean
shared_target_attention_mass_recall_mean
shared_target_stable_fraction_mean
```

如果某个 threshold 下能省下较多 selector，同时 mass recall 很高，说明 position sharing 有直接工程价值。

第二步看每层差异：

```text
shared_head_groups.csv
```

重点看哪些 layer 出现大 group。预期不是所有层都同样可共享，因为 Section 35 已经显示 layer 之间 top2 token 分布差异很大。

第三步看每个 target head 的最佳 donor：

```text
head_best_donors.csv
```

如果很多 target head 的 best donor recall 都高，但 greedy group 省不下太多 selector，说明共享关系可能是 many-to-one 不明显，或者 pairwise 高但不能组成稳定 group。

第四步看 query 依赖性：

```text
query_group_recall_by_threshold.csv
query_sharing_stability.csv
```

如果同一 threshold 下，不同 query token 的 `shared_target_top2_recall_mean` 方差很大，说明共享策略需要 query-aware gate。

## 5. 预期结论形态

这个实验不直接替代 PPL 实验，而是先做 selector-level diagnosis。

如果结果显示：

```text
top2_recall_mean 高
attention_mass_recall_mean 更高
query_group_recall_by_threshold 稳定
```

那么下一步可以做真正 sparse attention PPL：

```text
每个 group 只为 representative head 计算 top2 positions，
其他 member heads 复用 representative positions，
再比较 full/top2/shared-top2 PPL。
```

如果结果显示：

```text
aggregate 高，但 query-level 波动大
```

那么更合适的设计是：

```text
静态 head group + query-aware fallback
```

例如当当前 query 的共享 confidence 低时，target head 回退到自己的 selector。

## 6. 2026-07-03 服务器运行

服务器：

```text
fdong@10.176.37.31
GPU: RTX 3090
model: /home/fdong/hrj/prove/Qwen3-0.6B
```

### 6.1 All-token top2 position sharing

远端输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_war_4k_20260703_v1
```

本地已同步输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_war_4k_20260703_v1
```

阈值汇总：

| threshold | selectors with sharing | selectors saved | reduction | shared target count | top2 recall | mass recall | stable fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 293 | 155 / 448 | 34.60% | 155 | 0.573 | 0.941 | 0.767 |
| 0.6 | 395 | 53 / 448 | 11.83% | 53 | 0.652 | 0.955 | 0.728 |
| 0.7 | 438 | 10 / 448 | 2.23% | 10 | 0.737 | 0.985 | 0.717 |
| 0.8 | 448 | 0 / 448 | 0.00% | 0 | 1.000 | 1.000 | 1.000 |
| 0.9 | 448 | 0 / 448 | 0.00% | 0 | 1.000 | 1.000 | 1.000 |

最高 top2 recall 的有向 head pair：

| layer | donor -> target | top2 recall | mass recall | recall >= 0.8 fraction |
| ---: | --- | ---: | ---: | ---: |
| 6 | 6 -> 7 | 0.773 | 0.988 | 0.391 |
| 6 | 7 -> 6 | 0.773 | 0.841 | 0.391 |
| 15 | 4 -> 5 | 0.770 | 0.983 | 0.281 |
| 15 | 5 -> 4 | 0.770 | 0.982 | 0.281 |
| 4 | 0 -> 1 | 0.761 | 0.998 | 0.250 |
| 4 | 1 -> 0 | 0.761 | 0.981 | 0.250 |
| 13 | 12 -> 13 | 0.749 | 0.994 | 0.203 |
| 13 | 13 -> 12 | 0.749 | 0.990 | 0.203 |

阈值 0.7 下只得到 10 个可共享 target，大多是 pair：

```text
L6: 6 -> 7
L15: 4 -> 5
L4: 0 -> 1
L13: 12 -> 13
L17: 15 -> 14
L10: 14 -> 15
L18: 7 -> 6
L12: 4 -> 5
```

query-level 稳定性：

| threshold | query recall mean | min | p10 | p50 | p90 | mass mean | mass min | mass p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.573 | 0.511 | 0.541 | 0.568 | 0.610 | 0.941 | 0.901 | 0.927 |
| 0.6 | 0.652 | 0.589 | 0.624 | 0.647 | 0.691 | 0.955 | 0.918 | 0.943 |
| 0.7 | 0.737 | 0.676 | 0.707 | 0.733 | 0.772 | 0.985 | 0.947 | 0.973 |

解释：

```text
All-token position sharing 有一定 query-level 稳定性，p10 和 min 没有灾难性掉落。
但严格阈值下可共享的 head 很少。0.7 recall 只省 2.23% selector。
0.5 阈值能省 34.6% selector，但 set recall 只有 0.573；它的 mass recall 仍有 0.941，说明漏掉的很多 top2 token 不是主要 attention mass。
```

### 6.2 Remote-only top2 position sharing

远端输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_remote_war_4k_s64_r512_20260703_v1
```

本地已同步输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_remote_war_4k_s64_r512_20260703_v1
```

设置：

```text
exclude_sink_tokens = 64
exclude_recent_tokens = 512
```

阈值汇总：

| threshold | selectors with sharing | selectors saved | reduction | shared target count | top2 recall | mass recall | stable fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 405 | 43 / 448 | 9.60% | 43 | 0.595 | 0.950 | 0.725 |
| 0.6 | 432 | 16 / 448 | 3.57% | 16 | 0.700 | 1.074 | 0.767 |
| 0.7 | 439 | 9 / 448 | 2.01% | 9 | 0.749 | 1.090 | 0.663 |
| 0.8 | 446 | 2 / 448 | 0.45% | 2 | 0.818 | 1.082 | 0.625 |
| 0.9 | 448 | 0 / 448 | 0.00% | 0 | 1.000 | 1.000 | 1.000 |

最高 remote top2 recall 的有向 head pair：

| layer | donor -> target | top2 recall | mass recall | recall >= 0.8 fraction |
| ---: | --- | ---: | ---: | ---: |
| 13 | 13 -> 12 | 0.831 | 1.079 | 0.672 |
| 14 | 14 -> 15 | 0.806 | 1.086 | 0.578 |
| 10 | 15 -> 14 | 0.795 | 0.966 | 0.547 |
| 6 | 6 -> 7 | 0.781 | 0.989 | 0.422 |
| 6 | 7 -> 6 | 0.779 | 0.957 | 0.391 |
| 10 | 6 -> 1 | 0.715 | 1.728 | 0.429 |
| 16 | 8 -> 9 | 0.705 | 1.067 | 0.365 |
| 16 | 14 -> 15 | 0.705 | 0.935 | 0.156 |

query-level 稳定性：

| threshold | query recall mean | min | p10 | p50 | p90 | mass mean | mass min | mass p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.595 | 0.500 | 0.550 | 0.596 | 0.643 | 0.949 | 0.806 | 0.851 |
| 0.6 | 0.700 | 0.618 | 0.651 | 0.697 | 0.743 | 1.074 | 0.879 | 0.945 |
| 0.7 | 0.749 | 0.649 | 0.676 | 0.757 | 0.808 | 1.090 | 0.881 | 0.929 |
| 0.8 | 0.818 | 0.646 | 0.711 | 0.820 | 0.940 | 1.082 | 0.774 | 0.918 |

解释：

```text
排除 sink/recent 后，可共享 head 数显著减少。
0.5 阈值只省 43/448 个 selector，而 all-token 能省 155/448。
这说明 all-token 共享信号中有相当一部分来自 sink/recent position 结构。
```

但 remote-only 并不是完全没有共享：

```text
L13 H13 -> H12
L14 H14 -> H15
L10 H15 -> H14
L6 H6 -> H7
```

这些 pair 在 remote-only 下仍然达到 0.78-0.83 top2 recall。

值得注意的是，remote-only 的 mass recall 经常大于 1：

```text
donor positions 上的 target attention mass
>
target 自己 remote top2 positions 上的 attention mass
```

这是因为 remote-only 的分母只统计 target 自己被保留的 remote top2 selected positions；donor positions 可能覆盖 target top2 之外但 full attention mass 更高的 remote positions。因此 remote-only 的 mass recall 应解释为“复用 donor remote positions 的 attention mass 覆盖”，不应简单等同于 set recall。

## 7. 当前结论

1. Head position sharing 有信号，但不是大规模静态共享。

   在 all-token 设置下，放宽到 0.5 recall 可以省 34.6% selector，但严格到 0.7 recall 只能省 2.23%。这说明“多个 head 共用同一组 2% positions”不能直接全局套用。

2. 共享关系不是完全 query-dependent。

   query-level 表里 p10 和 min 都没有突然崩掉。例如 all-token threshold 0.7 的 query recall min 为 0.676，p10 为 0.707；remote-only threshold 0.7 的 min 为 0.649，p10 为 0.676。共享关系有一定稳定性，但高阈值下可共享 head 数量很小。

3. Sink/recent 是 all-token 共享的重要来源。

   all-token threshold 0.5 可以省 155 个 selector；remote-only 只省 43 个。排除 sink 和 recent 后，可共享空间明显收缩。

4. 更合理的方向不是所有 head 静态共享，而是：

```text
少数稳定 head-pair position sharing
+ sink/recent 单独处理
+ remote-only head-pair 白名单
+ query-aware fallback
```

5. 已完成第一版 PPL 验证。

   selector-level diagnosis 后，已经实现了一个 `sharedtop2` sparse attention mode：

```text
对白名单 pair/group：
  representative head 计算 top2 positions
  member head 复用 representative positions
其他 head：
  仍然计算自己的 top2 positions
```

优先测试三组：

```text
all-token threshold 0.7 groups
remote-only threshold 0.7 groups + sink/recent protect
all-token threshold 0.5 groups + query-aware fallback
```

## 8. PPL 与时间开销验证

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_head_position_sharing_ppl_server.sh
```

新增评测模式：

```text
sharedtop2t0p7attn   # all-token group threshold = 0.7
sharedtop2t0p6attn   # all-token group threshold = 0.6
sharedtop2t0p5attn   # all-token group threshold = 0.5
sharedtop2rt0p7attn  # remote-only group threshold = 0.7
sharedtop2rt0p6attn  # remote-only group threshold = 0.6
sharedtop2rt0p8attn  # remote-only group threshold = 0.8
```

实现细节：

```text
DISABLE_SPARSE_STATS=true 时，sharedtop2 不再先给所有 head 计算 top2 后再替换。
它只对每层 group 的 representative heads 做 topk selector，
然后用缓存好的 head index 把 representative mask 展开给 member heads。
```

### 8.1 Eval 128 tokens

设置：

```text
prefill_tokens = 4096
eval_tokens = 128
top_fraction = 0.02
chunk_size = eval_chunk_size = 64
reuse_prefill_cache = true
```

输出：

```text
all-token:
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_ppl_all_war_4k_eval128_20260703_v3

remote-only:
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_ppl_remote_war_4k_eval128_s64r512_20260703_v3
```

All-token：

| mode | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 32.2123 | -0.0413 | 0.080 | 0.07x |
| top2 | 32.2536 | 0.0000 | 1.113 | 1.00x |
| sharedtop2 t=0.7 | 32.3969 | +0.1433 | 1.599 | 1.44x |
| sharedtop2 t=0.6 | 31.7228 | -0.5308 | 1.519 | 1.36x |
| sharedtop2 t=0.5 | 34.2526 | +1.9989 | 1.517 | 1.36x |

Remote-only，sink=64、recent=512：

| mode | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 32.2123 | +0.3894 | 0.079 | 0.06x |
| top2 | 31.8229 | 0.0000 | 1.380 | 1.00x |
| sharedtop2r t=0.8 | 31.8397 | +0.0168 | 1.866 | 1.35x |
| sharedtop2r t=0.7 | 31.7877 | -0.0352 | 1.785 | 1.29x |
| sharedtop2r t=0.6 | 31.7684 | -0.0545 | 1.782 | 1.29x |

### 8.2 Eval 512 tokens

为了减少 128-token 固定开销的影响，又跑了一组 eval=512：

```text
all-token:
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_ppl_all_war_4k_eval512_20260703_v1

remote-only:
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_head_position_sharing_ppl_remote_war_4k_eval512_s64r512_20260703_v1
```

All-token：

| mode | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 23.4240 | +0.0461 | 0.308 | 0.07x |
| top2 | 23.3779 | 0.0000 | 4.327 | 1.00x |
| sharedtop2 t=0.7 | 23.4034 | +0.0254 | 5.969 | 1.38x |
| sharedtop2 t=0.6 | 23.5181 | +0.1402 | 5.898 | 1.36x |

Remote-only，sink=64、recent=512：

| mode | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 23.4240 | +0.0917 | 0.313 | 0.06x |
| top2 | 23.3323 | 0.0000 | 5.616 | 1.00x |
| sharedtop2r t=0.7 | 23.3296 | -0.0028 | 7.313 | 1.30x |
| sharedtop2r t=0.6 | 23.3834 | +0.0511 | 7.224 | 1.29x |

### 8.3 结论

PPL 结论：

```text
共享 position 的质量风险不大，尤其 remote-only t=0.7。
eval512 上 remote-only t=0.7 与 top2 基本一致：23.3296 vs 23.3323。
all-token t=0.7 也接近 top2：23.4034 vs 23.3779。
all-token t=0.5 在 eval128 上 PPL 明显变差，不建议直接用。
```

时间结论：

```text
当前 PyTorch eager 原型没有得到 wall-time 加速。
即使 sharedtop2 已经只对 representative heads 做 topk，
eval512 仍然比 top2 慢约 1.30x-1.38x。
```

原因：

```text
torch.topk 在 16 heads 上已经是高度向量化的小张量操作。
sharedtop2 省掉的 topk head 数不够大，尤其 threshold 0.6/0.7 下 selector reduction 很小。
额外的 index_select / mask expand / scatter / Python per-query 调度开销盖过了节省的 topk。
remote-only 本身可共享 selector 更少，所以更难在 eager 原型里体现速度收益。
```

因此当前策略的工程判断是：

```text
作为质量策略：remote-only t=0.7 可以继续保留，PPL 基本无损。
作为时间优化：当前 eager 原型不值得直接合入，必须做 fused selector/sparse attention，
或者把共享策略和更大粒度的 selector reuse 结合，才可能真的减少端到端时间。
```

## 9. 相邻 step 的 top2 position sharing

本节继续测试另一个方向：

```text
同一 layer/head 在相邻 query step 之间，是否可以共享 top2% historical token index。
```

定义：

```text
S(l,h,q) = layer l、head h、query token q 的真实 full-QK top2% historical positions

adjacent recall(q-1 -> q) =
  |S(l,h,q-1) intersect S(l,h,q)| / |S(l,h,q)|
```

另外输出 `old_history_top2_recall`：

```text
|S(q-1) intersect S(q)| / |S(q) 中在 q-1 时已经存在的 positions|
```

这个指标把“新增 token 导致前一步不可能选中”的影响剥离出来。

### 9.1 新增脚本

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_adjacent_step_position_sharing.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_adjacent_step_position_sharing_server.sh
```

默认服务器运行：

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_position_sharing_war_4k_20260703_v1 \
bash scripts/run_top2_adjacent_step_position_sharing_server.sh
```

Remote-only 版本：

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_position_sharing_remote_war_4k_s64_r512_20260703_v1 \
REMOTE_ONLY=true \
EXCLUDE_SINK_TOKENS=64 \
EXCLUDE_RECENT_TOKENS=512 \
bash scripts/run_top2_adjacent_step_position_sharing_server.sh
```

### 9.2 合并组定义

对每个 threshold、layer、head，按 query token 顺序做连续 run：

```text
1. run 的第一个 query token 作为 representative，计算一次 top2 selector。
2. 后续 step 如果 representative positions 对当前 step 的 recall >= threshold，
   则并入当前 run，复用 representative index。
3. 否则从当前 step 重新开一个 run，重新计算 selector。
```

因此：

```text
selectors_without_sharing = layer * head * observed_steps
selectors_with_sharing = 连续 run 数
selectors_saved = observed selector cases - run 数
```

输出文件：

```text
step_sharing_threshold_summary.csv
shared_step_groups.csv
layer_head_adjacent_step_stats.csv
query_adjacent_step_stability.csv
query_group_recall_by_threshold.csv
```

### 9.3 All-token 结果

设置：

```text
prefill_tokens = 4096
eval_tokens = 64 contiguous steps
layers = 28
heads = 16
top_fraction = 0.02
```

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_position_sharing_war_4k_20260703_v1
```

阈值汇总：

| threshold | selectors saved | reduction | mean group steps | max group steps | shared recall | shared mass recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 15035 / 28672 | 52.44% | 2.10 | 42 | 0.607 | 0.810 |
| 0.6 | 9890 / 28672 | 34.49% | 1.53 | 34 | 0.679 | 0.839 |
| 0.7 | 4482 / 28672 | 15.63% | 1.19 | 25 | 0.758 | 0.865 |
| 0.8 | 1195 / 28672 | 4.17% | 1.04 | 17 | 0.838 | 0.885 |
| 0.9 | 129 / 28672 | 0.45% | 1.00 | 8 | 0.924 | 0.903 |

Adjacent transition 本身的 query-level 稳定性：

```text
prev_to_current recall mean = 0.542
min = 0.381
p10 = 0.484
p50 = 0.535
p90 = 0.608
```

按 query 看 run-sharing 的稳定性：

| threshold | query reduction mean | p10 | p50 | p90 | shared recall mean | recall p10 | mass mean | mass p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.524 | 0.407 | 0.515 | 0.691 | 0.605 | 0.591 | 0.809 | 0.779 |
| 0.6 | 0.345 | 0.219 | 0.333 | 0.504 | 0.677 | 0.665 | 0.836 | 0.804 |
| 0.7 | 0.156 | 0.076 | 0.129 | 0.265 | 0.757 | 0.750 | 0.855 | 0.803 |
| 0.8 | 0.042 | 0.013 | 0.033 | 0.084 | 0.838 | 0.829 | 0.867 | 0.801 |
| 0.9 | 0.004 | 0.002 | 0.002 | 0.010 | 0.929 | 0.907 | 0.884 | 0.697 |

最稳定的 all-token layer/head：

| layer | head | adjacent recall | mass recall | recall >= 0.7 fraction |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 0.964 | 0.971 | 1.000 |
| 1 | 4 | 0.862 | 0.932 | 1.000 |
| 10 | 0 | 0.832 | 0.892 | 1.000 |
| 9 | 1 | 0.828 | 0.930 | 0.952 |
| 4 | 1 | 0.825 | 0.816 | 0.937 |
| 14 | 4 | 0.821 | 0.911 | 0.921 |
| 10 | 1 | 0.812 | 0.955 | 0.968 |
| 17 | 15 | 0.810 | 0.881 | 0.937 |

解释：

```text
相邻 step 的 all-token sharing 信号明显强于跨 head sharing。
threshold 0.7 能省 15.63% selector，而之前跨 head all-token threshold 0.7 只能省 2.23%。
但 all-token shared mass recall 只有 0.865 左右，说明直接复用仍有质量风险。
```

### 9.4 Remote-only 结果

设置：

```text
exclude_sink_tokens = 64
exclude_recent_tokens = 512
```

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_position_sharing_remote_war_4k_s64_r512_20260703_v1
```

阈值汇总：

| threshold | selectors saved | reduction | mean group steps | max group steps | shared recall | shared mass recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 8294 / 28145 | 29.47% | 1.42 | 60 | 0.644 | 1.019 |
| 0.6 | 4884 / 28145 | 17.35% | 1.21 | 37 | 0.734 | 1.126 |
| 0.7 | 2513 / 28145 | 8.93% | 1.10 | 25 | 0.824 | 1.278 |
| 0.8 | 1278 / 28145 | 4.54% | 1.05 | 21 | 0.905 | 1.490 |
| 0.9 | 578 / 28145 | 2.05% | 1.02 | 7 | 0.982 | 1.888 |

Remote-only 的 selector cases 是 28145 而不是 28672，因为有少量 layer/head/step 的 true top2 全落在 sink/recent 区域，过滤后 remote selected set 为空。

Adjacent transition 本身的 query-level 稳定性：

```text
prev_to_current recall mean = 0.365
min = 0.215
p10 = 0.291
p50 = 0.350
p90 = 0.464
```

按 query 看 run-sharing 的稳定性：

| threshold | query reduction mean | p10 | p50 | p90 | shared recall mean | recall p10 | mass mean | mass p10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.294 | 0.194 | 0.273 | 0.445 | 0.643 | 0.614 | 1.024 | 0.934 |
| 0.6 | 0.173 | 0.103 | 0.154 | 0.290 | 0.735 | 0.712 | 1.137 | 1.025 |
| 0.7 | 0.089 | 0.047 | 0.077 | 0.159 | 0.826 | 0.802 | 1.292 | 1.129 |
| 0.8 | 0.045 | 0.021 | 0.040 | 0.079 | 0.905 | 0.879 | 1.507 | 1.217 |
| 0.9 | 0.021 | 0.008 | 0.020 | 0.039 | 0.981 | 0.965 | 1.918 | 1.435 |

最稳定的 remote-only layer/head：

| layer | head | adjacent recall | mass recall | recall >= 0.7 fraction |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 14 | 0.781 | 0.927 | 0.698 |
| 13 | 12 | 0.767 | 0.905 | 0.778 |
| 13 | 13 | 0.761 | 0.900 | 0.730 |
| 10 | 15 | 0.725 | 0.877 | 0.603 |
| 10 | 1 | 0.708 | 1.053 | 0.565 |
| 5 | 12 | 0.683 | 0.934 | 0.508 |
| 26 | 10 | 0.681 | 0.908 | 0.492 |
| 10 | 6 | 0.678 | 0.900 | 0.492 |

解释：

```text
remote-only 的 adjacent set recall 均值比 all-token 低，但一旦通过阈值，shared mass recall 更高。
threshold 0.7 时可省 8.93% remote selector，shared recall 0.824，mass recall 1.278。
这比跨 head remote-only threshold 0.7 的 2.01% selector saving 明显更有工程价值。
```

### 9.5 当前判断

1. 相邻 step sharing 比跨 head sharing 更值得继续做。

```text
all-token t=0.7: 15.63% selector saving
remote-only t=0.7: 8.93% selector saving
```

而跨 head t=0.7 分别只有：

```text
all-token: 2.23%
remote-only: 2.01%
```

2. 直接 all-token 共享仍有质量风险。

```text
all-token t=0.7 shared mass recall = 0.865
all-token t=0.6 shared mass recall = 0.839
```

如果做 PPL，all-token 版本应优先加 fallback 或 mass/query-aware gate。

3. Remote-only 相邻 step sharing 是更干净的下一步。

```text
remote-only t=0.7:
  selector saving = 8.93%
  shared top2 recall = 0.824
  shared mass recall = 1.278
  query p10 saving = 4.7%
  query p10 shared recall = 0.802
```

这说明在 sink/recent 已保护的前提下，remote selector 可以跨相邻 step 做一部分复用。

4. 推荐下一步做 PPL 验证的优先级：

```text
第一优先级：
  remote-only adjacent-step sharing threshold 0.7
  sink=64, recent=512

第二优先级：
  remote-only threshold 0.8
  更保守，saving 4.54%，recall 0.905

第三优先级：
  all-token threshold 0.7 + fallback gate
```

## 10. 固定 stride 的 adjacent-step PPL 验证

根据第 9 节的 selector-level 结果，直接实现一个不依赖 oracle threshold 的运行时版本：

```text
adjtop2sKattn   = all-token adjacent-step top2 reuse, 每 K step refresh 一次 selector
adjtop2rsKattn  = remote-only adjacent-step top2 reuse, 每 K step refresh 一次 remote selector
```

实现位置：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_adjacent_step_reuse_ppl_server.sh
```

运行逻辑：

```text
1. refresh step：计算当前 query 的真实 top2 selector，并保存为 representative mask。
2. reuse step：跳过当前 step 的 topk selector，直接复用 representative mask。
3. remote-only 版本只复用 remote 区域；sink/recent 仍然全保留。
```

这是真实能减少 topk selector 次数的版本，不是 oracle threshold 版本。

### 10.1 sink=64, recent=512

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_reuse_ppl_war_4k_eval512_20260703_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 23.4240 | +0.0917 | 0.312 | 0.06x |
| top2 | - | 23.3323 | 0.0000 | 5.505 | 1.00x |
| adjtop2rs2attn | 0.500 | 24.3419 | +1.0096 | 5.953 | 1.08x |
| adjtop2rs3attn | 0.666 | 24.7352 | +1.4029 | 5.306 | 0.96x |
| adjtop2rs4attn | 0.750 | 24.7312 | +1.3988 | 5.029 | 0.91x |
| adjtop2s2attn | 0.500 | 24.3499 | +1.0176 | 5.372 | 0.98x |

结论：

```text
recent=512 时质量损失太大。
rs3/rs4 虽然略快，但 PPL 损失超过 1.4，不能作为可用方案。
```

### 10.2 sink=64, recent=1024

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_reuse_ppl_war_4k_eval512_s64r1024_20260703_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 23.4240 | -0.0708 | 0.313 | 0.06x |
| top2 | - | 23.4948 | 0.0000 | 5.576 | 1.00x |
| adjtop2rs2attn | 0.500 | 23.6810 | +0.1862 | 6.037 | 1.08x |
| adjtop2rs3attn | 0.666 | 24.0684 | +0.5736 | 5.396 | 0.97x |

结论：

```text
保护更大的 recent window 可以改善 PPL，但 rs2 没有速度收益，rs3 仍有明显 PPL 损失。
```

### 10.3 sink=64, recent=2048

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_reuse_ppl_war_4k_eval512_s64r2048_20260703_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 23.4240 | -0.1393 | 0.313 | 0.06x |
| top2 | - | 23.5633 | 0.0000 | 5.488 | 1.00x |
| adjtop2rs2attn | 0.500 | 23.5420 | -0.0213 | 5.923 | 1.08x |
| adjtop2rs3attn | 0.666 | 23.7465 | +0.1832 | 5.293 | 0.96x |

结论：

```text
recent=2048 时 rs2 的 PPL 接近 top2，但速度反而慢。
rs3 有 4% 左右速度收益，但 PPL 损失约 0.18。
这不是一个足够干净的 Pareto 改进。
```

### 10.4 工程判断

固定 stride adjacent-step reuse 目前不建议作为主线继续推进：

```text
速度：只在 stride >= 3 时略快于 top2 eager 原型。
质量：stride >= 3 的 PPL 损失明显；stride = 2 质量较好但没有速度收益。
```

这说明第 9 节的 selector-level mass recall 虽然看起来不错，但真实 PPL 对“固定复用上一组 remote top2 index”比较敏感。更值得继续的版本应该是：

```text
1. gated adjacent reuse：
   只在 query/head/layer confidence 足够高时复用，否则 refresh。

2. stable layer/head whitelist：
   只对 L10/L13 等 adjacent recall 高的 remote heads 做 reuse。

3. fused selector/sparse attention：
   当前 eager 原型中，减少 topk 调用的收益仍容易被 mask/index 开销吃掉。
```

当前已实现的固定 stride mode 可以保留作为 negative baseline 和后续 gated 版本的对照。

## 11. stable head whitelist adjacent-step reuse 验证

固定 stride 的问题是对所有 head 都强制复用上一组 remote top2 index，PPL 损失比较明显。因此进一步做一个更保守的版本：只对 selector 统计里相邻 step 稳定的 head 做 reuse，其余 head 仍然计算当前 step 的 top2。

实现位置：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_adjacent_step_reuse_ppl_server.sh
```

新增 mode：

```text
adjtop2rwlt0p7s2attn = remote-only whitelist, recall threshold 0.7, stride 2
adjtop2rwlt0p6s2attn = remote-only whitelist, recall threshold 0.6, stride 2
adjtop2rwlt0p5s2attn = remote-only whitelist, recall threshold 0.5, stride 2
adjtop2rwlt0p7s3attn = remote-only whitelist, recall threshold 0.7, stride 3
adjtop2rwlt0p6s3attn = remote-only whitelist, recall threshold 0.6, stride 3
adjtop2rwlt0p5s3attn = remote-only whitelist, recall threshold 0.5, stride 3
```

whitelist 来自 remote-only adjacent selector 统计：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_position_sharing_remote_war_4k_s64_r512_20260703_v1/layer_head_adjacent_step_stats.csv
```

判定规则：

```text
如果某个 layer/head 的 prev_to_current_top2_recall_mean >= threshold，
则这个 head 在 reuse step 复用上一组 remote top2 index；
否则仍然计算当前 step 的 remote top2。
sink/recent/self 保护逻辑保持不变。
```

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_adjacent_step_whitelist_ppl_war_4k_eval512_s64r512_20260703_v2
```

实验配置：

```text
model = Qwen3-0.6B
dataset = War and Peace
eval tokens = 512
context length = 4096
sink = 64
recent = 512
top fraction = 2%
```

| mode | effective reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | - | 23.4240 | +0.0917 | 0.325 | 0.06x |
| top2 | - | 23.3323 | 0.0000 | 5.556 | 1.00x |
| adjtop2rwlt0p7s2attn | 0.0056 | 23.3364 | +0.0041 | 10.748 | 1.93x |
| adjtop2rwlt0p6s2attn | 0.0112 | 23.3344 | +0.0021 | 10.267 | 1.85x |
| adjtop2rwlt0p5s2attn | 0.0458 | 23.2857 | -0.0467 | 10.321 | 1.86x |
| adjtop2rwlt0p7s3attn | 0.0074 | 23.3402 | +0.0078 | 10.962 | 1.97x |
| adjtop2rwlt0p6s3attn | 0.0149 | 23.3326 | +0.0003 | 10.979 | 1.98x |
| adjtop2rwlt0p5s3attn | 0.0610 | 23.3239 | -0.0084 | 11.063 | 1.99x |

结论：

```text
1. PPL 是好的：
   threshold 0.7/0.6 基本和 top2 持平，threshold 0.5 在这组 512 eval token 上也没有退化。

2. 速度是不好的：
   whitelist 版本比普通 top2 慢约 1.85x 到 1.99x。
   原因是有效复用比例太低，只有 0.56% 到 6.10%，同时 partial-head topk、mask 合并和 index 操作引入了额外 eager 开销。

3. 这说明“稳定 head 可以复用”这个质量假设成立，但当前 eager 实现不是速度优化。
```

顺手看了 layer-level 的稳定性，最好的层平均 recall 也只有约 0.45 左右，而且层内 head 差异很大：

```text
layer 10 mean recall ~= 0.468, heads >= 0.5: 5, heads >= 0.6: 4
layer 13 mean recall ~= 0.457, heads >= 0.5: 3, heads >= 0.6: 2
layer 5  mean recall ~= 0.448, heads >= 0.5: 4, heads >= 0.6: 1
```

因此“整层复用”暂时也不够干净。它能减少 partial-head overhead，但会把很多不稳定 head 一起复用，预计 PPL 风险比 head whitelist 更大。

当前判断：

```text
fixed stride adjacent reuse：速度略有机会，但 PPL 不稳。
stable head whitelist：PPL 很稳，但速度更慢。
cross-head sharing：稳定 pair 存在，但 savings 太小。
```

所以这条线到目前为止还没有得到“PPL 低、速度快、质量好”的直接 Pareto 方案。真正值得继续做的工作应该转向 fused selector/sparse attention，或者设计一个 coarse gate，使得运行时能跳过整块 selector 计算，而不是在 eager Python 里做 partial-head mask 拼接。

## 12. multi-step top2 position reuse 上限分析

进一步验证“选一次 top2% token，能不能跨很多 decode step 复用”。这里先跑 remote-only 版本，因为 sink/recent/self 已经由保护逻辑全保留，真正需要昂贵 topk selector 的是 remote 区域。

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_multistep_position_sharing.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_multistep_position_sharing_server.sh
```

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1
```

配置：

```text
model = Qwen3-0.6B
dataset = War and Peace
prefill = 4096
eval = 512
top fraction = 2%
remote-only = true
sink = 64
recent = 512
layers = 28
heads = 16
```

### 12.1 固定每 K step 选一次 top2%

这个表模拟最直接的运行时策略：

```text
K=2  表示每 2 step 选一次，后 1 step 复用。
K=8  表示每 8 step 选一次，后 7 step 复用。
K=64 表示每 64 step 选一次，后 63 step 复用。
```

| horizon K | selector reduction | mean top2 recall | mass recall | recall >= 0.5 | recall >= 0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 50.00% | 0.373 | 0.620 | 31.25% | 8.86% |
| 3 | 66.60% | 0.325 | 0.561 | 24.44% | 6.95% |
| 4 | 75.00% | 0.297 | 0.519 | 20.96% | 5.64% |
| 8 | 87.50% | 0.232 | 0.426 | 14.09% | 3.87% |
| 16 | 93.75% | 0.197 | 0.374 | 10.74% | 2.95% |
| 32 | 96.88% | 0.164 | 0.329 | 7.94% | 2.15% |
| 64 | 98.44% | 0.153 | 0.311 | 6.84% | 1.78% |

结论：

```text
固定 K 步复用可以省很多 selector，但平均 recall 掉得太快。
K=8 虽然能省 87.5% selector，但 mean recall 只有 0.232。
K=16/32/64 的 recall 更低，不像能直接保住 PPL。
```

### 12.2 精确 lag=d 的衰减

这个表看“用 d step 之前选出的 top2 位置，覆盖当前 step true top2 的程度”：

| lag | mean top2 recall | mass recall | recall >= 0.5 | recall >= 0.7 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.369 | 0.618 | 30.59% | 8.90% |
| 2 | 0.284 | 0.505 | 18.72% | 4.94% |
| 4 | 0.222 | 0.419 | 12.14% | 3.12% |
| 8 | 0.191 | 0.373 | 9.55% | 2.53% |
| 16 | 0.154 | 0.318 | 6.81% | 1.86% |
| 32 | 0.123 | 0.270 | 4.81% | 1.36% |
| 64 | 0.104 | 0.230 | 3.70% | 1.02% |

结论：

```text
跨 1 step 已经只有 0.37 recall。
跨 8 step 后平均 recall 约 0.19。
跨 64 step 后平均 recall 约 0.10。
```

### 12.3 是否存在少数稳定 head

存在，但数量很少。

按固定 horizon 的 per-layer/head mean recall 统计：

| horizon K | heads >= 0.5 recall | total selector saving if only these heads reuse | heads >= 0.4 recall | total selector saving if only these heads reuse |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 34 / 448 | 3.79% | 179 / 448 | 19.98% |
| 4 | 15 / 448 | 2.51% | 54 / 448 | 9.04% |
| 8 | 11 / 448 | 2.15% | 25 / 448 | 4.88% |
| 16 | 10 / 448 | 2.09% | 19 / 448 | 3.98% |
| 32 | 10 / 448 | 2.16% | 13 / 448 | 2.81% |
| 64 | 7 / 448 | 1.54% | 12 / 448 | 2.64% |

最稳定的一批 head 很集中：

```text
L10 H14
L10 H15
L13 H13
L13 H12
L10 H6
L10 H1
L25 H2
L26 H10
L8  H0
L8  H13
```

其中 `L10 H14` 是最强信号：

```text
fixed horizon K=8  mean recall ~= 0.758, mass recall ~= 0.916
fixed horizon K=16 mean recall ~= 0.759, mass recall ~= 0.931
fixed horizon K=32 mean recall ~= 0.685, mass recall ~= 0.849
fixed horizon K=64 mean recall ~= 0.743, mass recall ~= 0.919
```

但这类 head 太少。即使用 K=64，对 recall >= 0.5 的 7 个 head 做复用，总 selector saving 也只有约 1.54%。

### 12.4 block-level 稳定性

如果要求一个 K-step block 内所有复用 step 的 recall 都不低于阈值，稳定性更弱：

| horizon K | block min recall >= 0.5 | block min recall >= 0.7 |
| ---: | ---: | ---: |
| 2 | 31.25% | 8.86% |
| 4 | 6.64% | 1.11% |
| 8 | 1.81% | 0.22% |
| 16 | 0.75% | 0.10% |
| 32 | 0.25% | 0.00% |
| 64 | 0.11% | 0.00% |

这说明“连续很多步都稳定”的 head/block 很稀疏，不能只看 mean recall。

### 12.5 当前判断

```text
1. 全 head 固定 K-step 复用：
   selector saving 高，但 recall 太低，预计 PPL 会明显坏。

2. 只复用稳定 head：
   PPL 风险小一些，但稳定 head 数量太少，总 selector saving 只有 1% 到 4% 量级。

3. 放宽到 recall >= 0.3：
   K=8 可达到约 16.8% 总 selector saving，K=16 约 10.5%，
   但 mean recall 只有 0.3 级别，质量风险很高。
```

因此，“一次 top2% 选择跨很多步给大量 head 复用”这条路线目前不成立。更可行的方向不是固定跨多步复用同一组 index，而是做一个低成本 predictor/gate：只在少数非常稳定的 head 或非常确定的 query 段复用；否则刷新 selector。

## 13. multi-step reuse PPL 验证

根据第 12 节的 selector-level 结果，继续做真实 PPL 验证。这里仍然使用：

```text
model = Qwen3-0.6B
dataset = War and Peace
prefill = 4096
eval = 512
sink = 64
recent = 512
top fraction = 2%
remote-only reuse
```

### 13.1 全 head 固定 K-step 复用

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_fixed_reuse_ppl_war_4k_eval512_s64r512_20260704_v1
```

mode：

```text
adjtop2rs8attn
adjtop2rs16attn
adjtop2rs32attn
adjtop2rs64attn
top2
baseline
```

结果：

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjtop2rs8attn | 0.8750 | 25.8520 | +2.5196 | 4.920 | 0.86x |
| adjtop2rs16attn | 0.9375 | 25.7399 | +2.4076 | 4.615 | 0.81x |
| adjtop2rs32attn | 0.9688 | 27.0703 | +3.7380 | 4.483 | 0.79x |
| adjtop2rs64attn | 0.9844 | 26.0881 | +2.7558 | 4.438 | 0.78x |
| top2 | - | 23.3323 | 0.0000 | 5.703 | 1.00x |
| baseline | - | 23.4240 | +0.0917 | 0.318 | 0.06x |

结论：

```text
全 head 固定 K-step 复用确实能变快，K=64 大约是 top2 的 0.78x。
但是 PPL 损失非常大，K=8 已经 +2.52，K=32 达到 +3.74。
这说明 selector-level mean recall 低的问题会直接反映到 PPL，不能作为可用方案。
```

### 13.2 multi-step stable-head whitelist 复用

为了避免全 head 质量崩掉，进一步用第 12 节的 per-horizon stable head 做 whitelist。whitelist 的分数来自：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1/fixed_horizon_layer_head.csv
```

派生为：

```text
outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1/multistep_whitelists/fixed_horizon_8_layer_head_whitelist.csv
outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1/multistep_whitelists/fixed_horizon_16_layer_head_whitelist.csv
outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1/multistep_whitelists/fixed_horizon_32_layer_head_whitelist.csv
outputs/top2_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1/multistep_whitelists/fixed_horizon_64_layer_head_whitelist.csv
```

#### K=8 whitelist

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_whitelist_ppl_h8_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjtop2rwlt0p3s8attn | 0.1680 | 23.5529 | +0.2206 | 12.583 | 2.21x |
| adjtop2rwlt0p4s8attn | 0.0488 | 23.3976 | +0.0653 | 12.292 | 2.16x |
| adjtop2rwlt0p5s8attn | 0.0215 | 23.3733 | +0.0410 | 12.239 | 2.15x |
| top2 | - | 23.3323 | 0.0000 | 5.693 | 1.00x |
| baseline | - | 23.4240 | +0.0917 | 0.317 | 0.06x |

#### K=16 whitelist

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_whitelist_ppl_h16_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjtop2rwlt0p3s16attn | 0.1046 | 23.4165 | +0.0842 | 12.588 | 2.24x |
| adjtop2rwlt0p4s16attn | 0.0398 | 23.3825 | +0.0502 | 12.370 | 2.20x |
| adjtop2rwlt0p5s16attn | 0.0209 | 23.3779 | +0.0456 | 12.356 | 2.19x |
| top2 | - | 23.3323 | 0.0000 | 5.632 | 1.00x |
| baseline | - | 23.4240 | +0.0917 | 0.313 | 0.06x |

#### K=32 whitelist

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_whitelist_ppl_h32_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjtop2rwlt0p3s32attn | 0.0757 | 23.3921 | +0.0598 | 12.856 | 2.25x |
| adjtop2rwlt0p4s32attn | 0.0281 | 23.4315 | +0.0992 | 12.620 | 2.21x |
| adjtop2rwlt0p5s32attn | 0.0216 | 23.4105 | +0.0781 | 12.606 | 2.21x |
| top2 | - | 23.3323 | 0.0000 | 5.702 | 1.00x |
| baseline | - | 23.4240 | +0.0917 | 0.395 | 0.07x |

#### K=64 whitelist

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_multistep_whitelist_ppl_h64_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top2 | seconds | time vs top2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjtop2rwlt0p3s64attn | 0.0791 | 23.4245 | +0.0922 | 12.264 | 2.28x |
| adjtop2rwlt0p4s64attn | 0.0264 | 23.4221 | +0.0898 | 12.047 | 2.24x |
| adjtop2rwlt0p5s64attn | 0.0154 | 23.3935 | +0.0611 | 12.031 | 2.23x |
| top2 | - | 23.3323 | 0.0000 | 5.383 | 1.00x |
| baseline | - | 23.4240 | +0.0917 | 0.310 | 0.06x |

### 13.3 PPL 结论

```text
1. 全 head multi-step reuse：
   能省时间，但 PPL 损失太大，不可用。

2. stable-head whitelist：
   PPL 可以接近 top2，通常 delta 在 +0.04 到 +0.10。
   但是有效 reuse fraction 只有 1.5% 到 16.8%，而 eager partial-head 实现比 top2 慢 2.1x 到 2.3x。

3. 最接近可用的质量点：
   K=8, threshold=0.5: PPL +0.041, reuse 2.15%
   K=16, threshold=0.5: PPL +0.046, reuse 2.09%
   K=32, threshold=0.3: PPL +0.060, reuse 7.57%
```

工程判断：

```text
multi-step reuse 的质量/速度 tradeoff 不够好。
全 head 版本有速度没质量；whitelist 版本有质量没速度。
当前 eager 实现里，partial-head topk + mask merge 的开销超过了少量 selector saving。
```

因此这条路线如果继续，必须换成 fused 实现或非常便宜的 coarse gate。单纯在 PyTorch eager 里做 multi-step whitelist 复用，不能成为最终优化。

## 14. top10% position overlap 复测

为验证 top2% 的不稳定是否主要来自 selector 边界过窄，重新用 `top_fraction=0.10` 跑同样的 overlap 统计。这里脚本名仍然是 `top2...`，但实际参数已经改为 top10%。

输出：

```text
# same-step cross-head
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_head_position_sharing_war_4k_20260704_v1
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_head_position_sharing_remote_war_4k_s64_r512_20260704_v1

# adjacent-step
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_adjacent_step_position_sharing_war_4k_20260704_v1
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_adjacent_step_position_sharing_remote_war_4k_s64_r512_20260704_v1

# multi-step remote-only
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_multistep_position_sharing_remote_war_4k_eval512_s64_r512_20260704_v1
```

配置：

```text
model = Qwen3-0.6B
dataset = War and Peace
prefill = 4096
top fraction = 10%
same-step / adjacent-step eval = 64
multi-step eval = 512
remote-only sink = 64
remote-only recent = 512
```

### 14.1 same-step cross-head overlap

all-token：

| threshold | selectors saved | reduction | shared top10 recall | shared mass recall | max group size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 292 / 448 | 65.18% | 0.588 | 0.947 | 15 |
| 0.6 | 165 / 448 | 36.83% | 0.665 | 0.969 | 10 |
| 0.7 | 49 / 448 | 10.94% | 0.736 | 0.980 | 6 |
| 0.8 | 2 / 448 | 0.45% | 0.824 | 0.997 | 2 |

remote-only：

| threshold | selectors saved | reduction | shared top10 recall | shared mass recall | max group size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 184 / 448 | 41.07% | 0.584 | 0.794 | 10 |
| 0.6 | 87 / 448 | 19.42% | 0.664 | 0.870 | 7 |
| 0.7 | 27 / 448 | 6.03% | 0.746 | 0.929 | 2 |
| 0.8 | 2 / 448 | 0.45% | 0.842 | 0.989 | 2 |

top10% 下 cross-head sharing 明显比 top2% 强很多。all-token 在 threshold=0.5 时甚至能形成接近整层的大 group，例如：

```text
L12 representative H9: 15 heads
L10 representative H1: 15 heads
L13 representative H2: 14 heads
L17 representative H3: 13 heads
```

remote-only 也有中等规模 group：

```text
L17 representative H8: 10 heads
L24 representative H6: 9 heads
L12 representative H3: 9 heads
L13 representative H4: 9 heads
```

说明 top10% 的“关注区域”在 head 之间确实有共享结构；top2% 的稀疏尖峰则更 head-specific。

### 14.2 adjacent-step overlap

all-token contiguous group：

| threshold | selector reduction | shared top10 recall | shared mass recall | max group steps |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 78.36% | 0.621 | 0.784 | 64 |
| 0.6 | 61.27% | 0.686 | 0.833 | 64 |
| 0.7 | 35.89% | 0.754 | 0.877 | 64 |
| 0.8 | 9.71% | 0.829 | 0.915 | 17 |

remote-only contiguous group：

| threshold | selector reduction | shared top10 recall | shared mass recall | max group steps |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 62.84% | 0.619 | 0.837 | 64 |
| 0.6 | 42.27% | 0.686 | 0.894 | 64 |
| 0.7 | 20.13% | 0.758 | 0.943 | 30 |
| 0.8 | 4.62% | 0.836 | 0.982 | 17 |

per-layer/head mean adjacent recall：

| setting | mean recall | heads >= 0.5 | heads >= 0.6 | heads >= 0.7 | heads >= 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-token | 0.664 | 426 / 448 | 363 / 448 | 165 / 448 | 13 / 448 |
| remote-only | 0.580 | 373 / 448 | 199 / 448 | 24 / 448 | 1 / 448 |

remote-only 最稳定的 adjacent heads：

```text
L0  H13: recall 0.803, mass 0.926
L5  H13: recall 0.793, mass 0.946
L14 H4 : recall 0.782, mass 0.954
L10 H14: recall 0.771, mass 0.955
L10 H1 : recall 0.761, mass 0.949
L5  H12: recall 0.761, mass 0.943
L5  H3 : recall 0.759, mass 0.928
L4  H1 : recall 0.752, mass 0.931
```

remote-only 最稳定的层：

```text
L10 mean 0.666
L5  mean 0.652
L17 mean 0.642
L15 mean 0.640
L12 mean 0.627
L13 mean 0.626
```

结论：相邻 step 的 top10% overlap 很强。top2% 的低 overlap 并不意味着语义完全变了，而是 exact top2% token selector 太尖锐。

### 14.3 multi-step remote-only overlap

固定每 K step 选一次 top10%，其余 step 复用：

| horizon K | selector reduction | mean top10 recall | mass recall | recall >= 0.5 | recall >= 0.7 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 50.00% | 0.584 | 0.792 | 73.73% | 23.04% |
| 3 | 66.60% | 0.540 | 0.740 | 62.89% | 16.70% |
| 4 | 75.00% | 0.516 | 0.711 | 57.30% | 13.53% |
| 8 | 87.50% | 0.452 | 0.630 | 41.71% | 7.86% |
| 16 | 93.75% | 0.414 | 0.578 | 33.33% | 5.28% |
| 32 | 96.88% | 0.374 | 0.522 | 25.37% | 3.14% |
| 64 | 98.44% | 0.364 | 0.507 | 22.26% | 2.24% |

精确 lag=d 衰减：

| lag | mean top10 recall | mass recall | recall >= 0.5 | recall >= 0.7 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.580 | 0.787 | 72.64% | 22.50% |
| 2 | 0.504 | 0.698 | 54.47% | 11.17% |
| 4 | 0.441 | 0.616 | 39.70% | 5.52% |
| 8 | 0.406 | 0.570 | 31.17% | 3.85% |
| 16 | 0.361 | 0.505 | 21.90% | 2.02% |
| 32 | 0.318 | 0.444 | 14.23% | 1.03% |
| 64 | 0.289 | 0.400 | 10.25% | 0.56% |

per-head 的 multi-step 稳定性也明显强于 top2%：

| horizon K | heads >= 0.5 | heads >= 0.6 | heads >= 0.7 |
| ---: | ---: | ---: | ---: |
| 2 | 382 / 448 | 211 / 448 | 24 / 448 |
| 4 | 286 / 448 | 71 / 448 | 8 / 448 |
| 8 | 135 / 448 | 25 / 448 | 2 / 448 |
| 16 | 83 / 448 | 20 / 448 | 2 / 448 |
| 32 | 49 / 448 | 11 / 448 | 1 / 448 |
| 64 | 43 / 448 | 7 / 448 | 1 / 448 |

top10% 下 multi-step 的强 head 不再只有 L10/L13，前层也出现稳定 head：

```text
K=8 top heads:
L0 H13, L5 H13, L0 H11, L10 H1, L5 H3, L2 H3, L14 H4, L10 H6

K=32 top heads:
L0 H13, L10 H1, L0 H11, L10 H6, L25 H2, L2 H3, L26 H10, L5 H3
```

### 14.4 top10% 结论

```text
1. top10% 的 adjacent-step overlap 很高：
   all-token mean adjacent recall = 0.664
   remote-only mean adjacent recall = 0.580

2. top10% 的 same-step cross-head sharing 也明显增强：
   all-token threshold=0.6 可以省 36.8% selector，mass recall 0.969
   remote-only threshold=0.6 可以省 19.4% selector，mass recall 0.870

3. 跨多 step 时仍会衰减，但 top10% 比 top2% 慢得多：
   remote K=8 mean recall = 0.452, mass recall = 0.630
   remote K=16 mean recall = 0.414, mass recall = 0.578
   remote K=64 mean recall = 0.364, mass recall = 0.507
```

这说明前面观察到的 top2% 不稳定，主要是“精确尖峰 token 集合”不稳定；扩到 top10% 后，相邻 step 和 head 之间确实共享了很多关注区域。后续如果要做工程优化，可以考虑两级策略：

```text
1. 低成本地预测/复用 top10% candidate region；
2. 在这个候选区域内部再精排/截断到 top2%。
```

这样可能比直接预测 exact top2% index 更稳定。

## 15. top10% reuse PPL 和测速

根据第 14 节的 overlap 结果，进一步跑真实 PPL 和 wall-clock time。注意这里 evaluator 的 mode 名仍然叫 `top2`/`adjtop2...`，但实际 `top_fraction=0.10`，所以它们表示 top10% selector。

配置：

```text
model = Qwen3-0.6B
dataset = War and Peace
prefill = 4096
eval = 512
sink = 64
recent = 512
top_fraction = 0.10
reuse_prefill_cache = true
```

### 15.1 top10 temporal reuse

remote-only 输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_ppl_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top10 | seconds | time vs top10 | time vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 selector | - | 23.1465 | 0.0000 | 5.535 | 1.00x | 17.99x |
| full attention baseline | - | 23.4240 | +0.2776 | 0.308 | 0.06x | 1.00x |
| adjtop2rs2attn | 0.5000 | 23.5596 | +0.4132 | 6.054 | 1.09x | 19.67x |
| adjtop2rs4attn | 0.7500 | 23.6602 | +0.5138 | 5.133 | 0.93x | 16.68x |
| adjtop2rs8attn | 0.8750 | 24.5593 | +1.4128 | 4.723 | 0.85x | 15.35x |
| adjtop2rs16attn | 0.9375 | 24.6642 | +1.5177 | 4.485 | 0.81x | 14.57x |
| adjtop2rs32attn | 0.9688 | 25.1161 | +1.9697 | 4.393 | 0.79x | 14.27x |
| adjtop2rs64attn | 0.9844 | 24.9444 | +1.7979 | 4.347 | 0.79x | 14.13x |

all-token 输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_alltoken_ppl_war_4k_eval512_s64r512_20260704_v1
```

| mode | reuse fraction | PPL | delta vs top10 | seconds | time vs top10 | time vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 selector | - | 23.1465 | 0.0000 | 5.567 | 1.00x | 17.73x |
| full attention baseline | - | 23.4240 | +0.2776 | 0.314 | 0.06x | 1.00x |
| adjtop2s2attn | 0.5000 | 23.5664 | +0.4199 | 5.588 | 1.00x | 17.79x |
| adjtop2s4attn | 0.7500 | 23.6594 | +0.5129 | 4.904 | 0.88x | 15.62x |
| adjtop2s8attn | 0.8750 | 24.5593 | +1.4128 | 4.597 | 0.83x | 14.64x |
| adjtop2s16attn | 0.9375 | 24.6642 | +1.5177 | 4.424 | 0.79x | 14.09x |

结论：

```text
top10 temporal reuse 可以相对普通 top10 selector 稍微加速：
  K=4 约 0.88x 到 0.93x
  K=8 约 0.83x 到 0.85x
  K=16 约 0.79x 到 0.81x

但 PPL 退化明显：
  K=2 已经 +0.41
  K=4 约 +0.51
  K=8 约 +1.41

所以即便 top10 overlap 比 top2 稳定，直接固定多步复用 top10 index 仍然不够好。
```

### 15.2 top10 cross-head sharing

remote-only 输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_cross_head_sharing_ppl_war_4k_eval512_s64r512_20260704_v1
```

| mode | group threshold | PPL | delta vs top10 | seconds | time vs top10 | time vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 selector | - | 23.1465 | 0.0000 | 5.420 | 1.00x | 17.49x |
| full attention baseline | - | 23.4240 | +0.2776 | 0.310 | 0.06x | 1.00x |
| sharedtop2rt0p5attn | 0.5 | 23.1485 | +0.0020 | 7.132 | 1.32x | 23.01x |
| sharedtop2rt0p6attn | 0.6 | 23.2722 | +0.1257 | 7.047 | 1.30x | 22.74x |
| sharedtop2rt0p7attn | 0.7 | 23.1171 | -0.0293 | 7.042 | 1.30x | 22.72x |

all-token 输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_cross_head_sharing_alltoken_ppl_war_4k_eval512_s64r512_20260704_v1
```

| mode | group threshold | PPL | delta vs top10 | seconds | time vs top10 | time vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top10 selector | - | 23.1465 | 0.0000 | 5.651 | 1.00x | 16.83x |
| full attention baseline | - | 23.4240 | +0.2776 | 0.336 | 0.06x | 1.00x |
| sharedtop2t0p5attn | 0.5 | 23.6644 | +0.5179 | 7.351 | 1.30x | 21.89x |
| sharedtop2t0p6attn | 0.6 | 23.1498 | +0.0033 | 7.277 | 1.29x | 21.67x |
| sharedtop2t0p7attn | 0.7 | 23.1041 | -0.0424 | 7.267 | 1.29x | 21.64x |

结论：

```text
cross-head top10 sharing 的 PPL 很好：
  remote t=0.5 基本无损，+0.002
  all-token t=0.6 基本无损，+0.003
  t=0.7 在这组 eval 上甚至略低于 top10

但是当前 eager 实现没有速度收益，反而慢约 1.3x。
原因是虽然 selector 数减少了，但 Python/PyTorch 里的 group index、gather/scatter、mask merge 开销更大。
```

### 15.3 工程判断

```text
top10 selector 本身质量很好：
  PPL = 23.1465
  full attention baseline PPL = 23.4240

但当前 eager top10 selector 很慢：
  top10 selector = 5.4s 到 5.6s
  full attention baseline = 0.31s 到 0.34s
```

这组实验说明：

```text
1. top10% region 是一个更稳定的 candidate region；
2. 固定多步复用 top10 exact index 仍会伤 PPL；
3. cross-head 共享 top10 region 的 PPL 风险很小；
4. 但要拿速度，必须把 selector/group sharing 做成 fused kernel 或者真正低成本的候选区域生成。
```

因此下一步更合理的实现方向是：

```text
先用低成本方法得到 top10% candidate region，
再在 candidate region 内做 top2% 精排。
```

直接在 eager 里复用 top10 index 不能作为最终速度优化，但 top10 region 作为稳定候选集是有价值的。

## 16. top10 temporal reuse 的真实 sparse gather 修正测速

第 15 节的 temporal reuse 计时仍然走了 full-score + mask 路径：先计算全量 `QK^T`，再把未选中的 token mask 掉。因此那组 wall-clock 只能说明 selector 逻辑和 PPL，不能代表真实 sparse attention 速度。

修正代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
```

修正点：

```text
adjtop2rsKattn 的 reuse step 不再构造 full scores。
当 eval_chunk_size=1 且当前 step 可复用上一组 remote top10 index 时：
  1. 从 AdjacentTop2ReuseState 取上一组 remote keep mask；
  2. 转为 selected_history_indices；
  3. 拼接 sink/recent/self；
  4. gather selected K/V；
  5. 只在 selected K/V 上做 QK、softmax、V reduce。

refresh step 仍然需要 full scores，因为它要重新选 top10。
```

注意：这仍然是 PyTorch eager gather sparse，不是 fused sparse decode kernel。

### 16.1 smoke: eval=64

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_sparse_reuse_smoke_eval64_s64r512_20260704_v1
```

| mode | PPL | seconds |
| --- | ---: | ---: |
| adjtop2rs16attn | 33.5354 | 3.920 |
| top10 selector | 33.5980 | 3.040 |
| full attention baseline | 33.7231 | 2.162 |

CUDA final kernel smoke：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_sparse_reuse_cuda_smoke_eval64_s64r512_20260704_v1
```

| mode | PPL | seconds |
| --- | ---: | ---: |
| adjtop2rs16attn | 33.5861 | 3.978 |
| top10 selector | 33.5980 | 3.055 |
| full attention baseline | 33.7231 | 2.174 |

`summary.json` 确认 `qabs_cuda_final_kernel=true`，但这组没有带来速度收益。

### 16.2 full eval=512

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_sparse_reuse_ppl_eval512_s64r512_20260704_v1
```

配置：

```text
top_fraction = 0.10
eval_chunk_size = 1
sink = 64
recent = 512
```

| mode | reuse fraction | PPL | delta vs top10 | seconds | time vs top10 | time vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adjtop2rs16attn | 0.9375 | 24.7540 | +1.5910 | 30.188 | 1.25x | 1.77x |
| adjtop2rs64attn | 0.9844 | 25.2296 | +2.0666 | 30.868 | 1.28x | 1.81x |
| top10 selector | - | 23.1631 | 0.0000 | 24.095 | 1.00x | 1.41x |
| full attention baseline | - | 23.4175 | +0.2544 | 17.087 | 0.71x | 1.00x |

### 16.3 修正后的结论

```text
之前的 mask 版本测速不能作为 sparse attention 速度结论。
修正成真实 gather sparse 后，reuse step 确实没有 full QK，但 PyTorch eager 仍然更慢。
```

原因：

```text
1. selected token 并不小：
   top10 remote 约 410 tokens，加 sink 64、recent 512、self 1，
   每 head 每 step 实际还是接近 987 tokens。

2. gather K/V 是非连续访问：
   对每层每 head 做 variable-index gather，内存访问不如 dense attention 连续。

3. 小 matmul/softmax 太碎：
   每 token、每层都做很多小 kernel，GPU 利用率低。

4. refresh step 仍然要 full scores：
   K=16 仍有 1/16 step 需要 full QK + topk。

5. full attention baseline 走的是高度优化的 dense matmul/softmax 路径：
   即使算更多 token，kernel launch 和内存访问更规整。
```

所以当前真实结论是：

```text
top10 temporal reuse 在 PyTorch eager gather sparse 下不可用：
  PPL 变差；
  速度也比 top10 selector 和 full attention baseline 都慢。
```

要真正验证这条路线的速度上限，需要实现 fused sparse decode kernel：

```text
输入：每 layer/head 的 selected indices
内核内完成：gather K/V + QK + masked softmax + V reduce
避免：Python loop、torch.gather 中间张量、小 matmul、小 softmax
```

没有 fused kernel 的情况下，继续调 PyTorch eager sparse 版本意义不大。

## 17. Section65 口径的 attention/KV subsystem 测速

按 `ymluo/doc/section65_speed_benchmark_protocol.md` 重新做测速。新的测速脚本只测 attention/KV 子系统，不再使用 HF generate 或 PPL evaluator 的整段 seconds 作为速度结论。

代码：
```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/benchmark_top10_temporal_reuse_attention_kv.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top10_temporal_reuse_attention_kv_benchmark_server.sh
```

配置：
```text
GPU: RTX 3090
torch: 2.4.0
model-shape: 28 layers, 16 heads, head_dim=64
full_kv_len: 4097
history_count: 4096
top_fraction: 0.10
top_count: 410
sink/recent/self: 64 / 512 / 1
active_kv_len: 987
dtype: float16
```

这个 benchmark 的单位是单层测量，然后乘以 28 层，得到整模型 attention/KV subsystem 的估算时间。它排除了 MLP、lm_head、tokenizer、HF forward 和 Python decode loop。

### 17.1 PyTorch gather-compact sparse

输出：
```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_attention_kv_section65_gather_20260705_v2
```

单层原始计时：
```text
full attention over 4097 KV: 0.1446 ms
refresh scoring QK over history: 0.0646 ms
top10 topk: 0.0238 ms
gather selected K/V: 0.0388 ms
attention over selected 987 KV: 0.1366 ms
```

按 28 层后的 section65 核心表：

| method | steps | scoring ms | topk ms | gather ms | attention ms | total ms | overhead share | speedup vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_attention | 16 | 0.000 | 0.000 | 0.000 | 64.760 | 64.760 | 0.000 | 1.000x |
| top10_temporal_reuse_k16_gather_compact | 16 | 1.807 | 0.667 | 17.373 | 61.188 | 81.036 | 0.245 | 0.799x |
| full_attention | 64 | 0.000 | 0.000 | 0.000 | 259.039 | 259.039 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 64 | 1.807 | 0.667 | 69.492 | 244.753 | 316.720 | 0.227 | 0.818x |
| full_attention | 1024 | 0.000 | 0.000 | 0.000 | 4144.622 | 4144.622 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 1024 | 28.920 | 10.673 | 1111.868 | 3916.054 | 5067.514 | 0.227 | 0.818x |

结论：
```text
topk 本身不是主瓶颈。
K=64 时，1024 steps 只 refresh 16 次，scoring+topk 合计只有 39.6 ms。
真正拖慢的是每步 gather selected K/V，以及 selected attention 没有明显快过 full attention。
active KV 从 4097 降到 987，但 PyTorch 的小 matmul/softmax 路径没有吃到线性收益。
```

### 17.2 fused indexed sparse attention

输出：
```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_attention_kv_section65_indexed_20260705_v2
```

这个版本使用已有 `qabs_cuda_kernels.final_attention`，不单独 compact K/V，因此 section65 表里 `gather_time=0`，随机 index 读 KV 的成本计入 `attention_time`。

单层原始计时：
```text
full attention over 4097 KV: 0.1427 ms
refresh scoring QK over history: 0.0638 ms
top10 topk: 0.0255 ms
indexed sparse attention over 987 KV: 0.3368 ms
```

按 28 层后的 section65 核心表：

| method | steps | scoring ms | topk ms | gather ms | attention ms | total ms | overhead share | speedup vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_attention | 16 | 0.000 | 0.000 | 0.000 | 63.909 | 63.909 | 0.000 | 1.000x |
| top10_temporal_reuse_k16_indexed_attention | 16 | 1.785 | 0.713 | 0.000 | 150.900 | 153.398 | 0.016 | 0.417x |
| full_attention | 64 | 0.000 | 0.000 | 0.000 | 255.635 | 255.635 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_indexed_attention | 64 | 1.785 | 0.713 | 0.000 | 603.598 | 606.097 | 0.004 | 0.422x |
| full_attention | 1024 | 0.000 | 0.000 | 0.000 | 4090.159 | 4090.159 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_indexed_attention | 1024 | 28.564 | 11.414 | 0.000 | 9657.574 | 9697.552 | 0.004 | 0.422x |

结论：
```text
indexed sparse kernel 证明了另一个问题：
当 gather 被移除后，overhead share 已经很低，K=64/1024 steps 只有 0.4%。
但当前 indexed attention kernel 自身比 full attention 慢约 2.36x。
所以现在不是“多久 refresh 一次 top10”的问题，而是 sparse attention kernel 没有优化到能打过 dense attention。
```

### 17.3 对当前路线的判断

这次按 section65 口径后，结论比 PPL seconds 更清楚：

```text
1. top10 refresh/topk 可以被 K=16/K=64 摊销掉。
2. 选一次 top10 复用很多步，在 overhead 口径上是成立的。
3. 但速度收益没有出现，因为 active attention 路径还没有快过 dense full attention。
4. PyTorch gather-compact 的问题是每步 gather 太重。
5. 当前 fused indexed kernel 的问题是 kernel 实现太朴素，随机访存和串行 softmax/reduce 太慢。
```

因此下一步如果目标是速度，应该停止继续调 mask/eager 版本，转向真正的 sparse decode kernel 优化：

```text
按 head/block 打包 selected index
减少每 head 一个小 kernel 的 launch/同步开销
让 selected K/V 访问更连续，或者按 page/block 而不是 token index 做选择
softmax + V reduce 必须 fused
否则 top10/top2 选得再少，也很难比 4k dense attention 快
```

### 17.4 修正：temporal reuse 不应该每步重复 gather

上面 17.1 的 gather-compact 表偏悲观：如果 K=64 表示一次 top10 选择给后续 64 步共享，那么 selected remote K/V 应该在 refresh 时 compact 一次，然后作为 compact cache 给这组 decode steps 复用；reuse step 不应该再次支付完整 gather。

已修正脚本：
```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/benchmark_top10_temporal_reuse_attention_kv.py
```

新的输出：
```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_attention_kv_section65_gather_once_20260705_v3
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top10_temporal_reuse_attention_kv_section65_indexed_gather_once_20260705_v3
```

修正后的 gather-compact 口径：

| method | steps | scoring ms | topk ms | gather ms | attention ms | total ms | overhead share | speedup vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_attention | 64 | 0.000 | 0.000 | 0.000 | 342.734 | 342.734 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 64 | 2.385 | 0.617 | 1.599 | 324.842 | 329.443 | 0.014 | 1.040x |
| full_attention | 1024 | 0.000 | 0.000 | 0.000 | 5483.738 | 5483.738 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 1024 | 38.154 | 9.870 | 25.587 | 5197.476 | 5271.088 | 0.014 | 1.040x |

这说明：
```text
1. 用户指出的问题是对的：K 步共享时不应每步重新 gather。
2. 修正后 K=64 的 overhead share 只有 1.4%，scoring/topk/gather 基本已经摊销掉。
3. 当前整体只有约 1.04x，是因为 selected attention over 987 KV 只比 full attention over 4097 KV 快约 5.2%。
4. 所以下一个瓶颈不是 topk，也不是 gather，而是 selected attention 实现本身没有按 active KV 缩小获得应有收益。
```

indexed sparse kernel 的结论不变：它没有显式 gather，但当前 kernel 自身仍慢于 dense full attention：

| method | steps | scoring ms | topk ms | gather ms | attention ms | total ms | speedup vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_attention | 1024 | 0.000 | 0.000 | 0.000 | 4132.291 | 4132.291 | 1.000x |
| top10_temporal_reuse_k64_indexed_attention | 1024 | 28.844 | 11.414 | 0.000 | 10267.237 | 10307.495 | 0.401x |

### 17.5 核心方法测速 repeat=1000

这次只比较 full attention 和 top10 temporal reuse 在 attention/KV 子系统里的差异部分，不包含 MLP、lm_head、tokenizer、HF forward、Python decode loop 等共同成本。

输出：
```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/core_method_attention_kv_top10_reuse_4k_repeat1000_20260705_v2
```

配置：
```text
full_kv_len = 4097
history_count = 4096
top_fraction = 0.10
top_count = 410
sink/recent/self = 64 / 512 / 1
active_kv_len = 987
layers = 28
heads = 16
head_dim = 64
repeat = 1000
```

单层计时：
```text
full attention over 4097 KV = 0.1482 ms
refresh scoring QK over history = 0.0640 ms
top10 topk = 0.0246 ms
compact selected K/V once = 0.0384 ms
selected attention over 987 KV = 0.1355 ms
```

按 28 层后的核心比较：

| method | steps | scoring ms | topk ms | gather ms | attention ms | total ms | overhead share | speedup vs full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_attention | 16 | 0.000 | 0.000 | 0.000 | 66.401 | 66.401 | 0.000 | 1.000x |
| top10_temporal_reuse_k16_gather_compact | 16 | 1.793 | 0.688 | 1.076 | 60.695 | 64.253 | 0.055 | 1.033x |
| full_attention | 64 | 0.000 | 0.000 | 0.000 | 265.603 | 265.603 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 64 | 1.793 | 0.688 | 1.076 | 242.781 | 246.338 | 0.014 | 1.078x |
| full_attention | 1024 | 0.000 | 0.000 | 0.000 | 4249.644 | 4249.644 | 0.000 | 1.000x |
| top10_temporal_reuse_k64_gather_compact | 1024 | 28.691 | 11.005 | 17.222 | 3884.492 | 3941.410 | 0.014 | 1.078x |

结论：
```text
K=64 共享后，scoring/topk/compact 的 overhead share 只有 1.4%。
核心 attention/KV 子系统有 1.078x 加速。
加速不大的原因不是 topk 或 gather，而是 selected attention over 987 KV 只比 full attention over 4097 KV 快约 8.6%。
```
