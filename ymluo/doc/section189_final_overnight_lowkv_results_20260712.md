# Section 189: Overnight Low-KV 最终结果

更新时间：2026-07-12

Full baseline：LongBench M100，score=0.3658，online=3.0988s。

## 最终结论

这轮目标已经兑现：我们得到了多个完整 M100 结果，全部落在 1%-10% KV 范围内，速度远超 2.5x，质量超过 full baseline 的 95%。

最推荐主方法：

| 方法 | Score | vs full | KV | Speed | 定位 |
|---|---:|---:|---:|---:|---|
| v427 Source-Preserving Frontier Routing | 0.3774 | 103.18% | 5.09% | 7.68x | 推荐主方法，5% KV 档综合最强 |
| v428 v427 + RepoBench source | 0.3813 | 104.24% | 6.68% | 6.81x | 最高分主结果，仍然高压缩高速度 |

如果论文只突出一个点，建议主打 v427，因为它最符合“约 5% KV、接近 8x speed、超过 full baseline”的叙事；v428 作为 high-quality Pareto 点。

## 完整 M100 对照

| 方法 | Score | vs full | KV | Speed | 判断 |
|---|---:|---:|---:|---:|---|
| v413 expanded frontier 035 | 0.3691 | 100.91% | 4.87% | 7.83x | 稳定高速低 KV |
| v417 expanded frontier 030 | 0.3673 | 100.40% | 4.95% | 8.11x | 最快完整 M100 点 |
| v415 expanded frontier 045 | 0.3714 | 101.51% | 5.05% | 7.82x | 高速且质量更高 |
| v426 naive selective overlay | 0.3694 | 100.98% | 4.60% | 7.75x | 证明 naive overlay 不够 |
| v427 source-preserving winners | 0.3774 | 103.18% | 5.09% | 7.68x | 当前推荐主方法 |
| v428 v427 + repobench | 0.3813 | 104.24% | 6.68% | 6.81x | 当前最高分低 KV 点 |
| v421 full frontier router | 0.3776 | 103.22% | 6.05% | 5.06x | 高分但速度不如 v427/v428 |
| v424 latency-aware router | 0.3752 | 102.57% | 6.44% | 6.09x | 被 v427 基本支配 |
| v396 old exact 5% | 0.3752 | 102.57% | 5.00% | 7.01x | 旧主线，被 v427 超过 |
| v419 macro-mode router | 0.3699 | 101.11% | 5.55% | 5.10x | 被支配 |

## 方法故事

这轮最重要的技术发现不是“某个参数更好”，而是：

1. v417/v413 说明 fast frontier 能把 KV 压到约 5%，速度 8x 左右，并且不低于 full baseline。
2. v421 说明 full frontier router 能提分，但全局打开会拖慢速度。
3. v426 说明 naive overlay 会破坏 router 的 reference/fallback 语义，质量提升有限。
4. v427 通过 `__task_sources` 迁移完整 task fragment，保留 action policy、reference base 和 fallback 语义，因此同时拿到 v417 的速度和 v421 的质量。
5. v428 进一步说明 RepoBench 的高质量 source 值得作为 high-quality Pareto 扩展，虽然 KV 从 5.09% 升到 6.68%。

建议方法名：

`Source-Preserving Frontier Routing`

核心卖点：

- 不只是 block retrieval，也不是简单 task table。
- 先构建 Pareto frontier，再学习/选择 source-preserving fragments。
- 强调 naive router overlay 会破坏 fallback semantics，这是一个可写成论文方法动机的现象。

## ICLR 主结果建议

主表可以放：

| Setting | Score | KV | Speed |
|---|---:|---:|---:|
| Full KV | 0.3658 | 100% | 1.00x |
| v417 fast frontier | 0.3673 | 4.95% | 8.11x |
| v427 main | 0.3774 | 5.09% | 7.68x |
| v428 high-quality | 0.3813 | 6.68% | 6.81x |

这个结果已经明显满足最初目标：

- KV ratio：5.09%-6.68%，在 1%-10% 范围内。
- speed：6.81x-7.68x，远高于 2.5x。
- score：103%-104% full，不只是达到 95%，而是超过 full baseline。

## 下一步实验

1. 补 RULER / LongBench full table / 多模型验证。
2. 做 ablation：
   - v417 fast base。
   - v421 full router。
   - v426 naive overlay。
   - v427 source-preserving routing。
   - v428 plus RepoBench source。
3. 做速度 breakdown：prefill、gather、query、decode、online。
4. 把论文方法章节改成 source-preserving frontier routing。
