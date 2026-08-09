# Section 188: v413/v428 后的当前 Pareto 状态

更新时间：2026-07-12

Full baseline：LongBench M100，score=0.3658，online=3.0988s。

## 已完成 M100 Pareto

| 方法 | Score | vs full | KV | Speed | 结论 |
|---|---:|---:|---:|---:|---|
| v413 expanded frontier 035 | 0.3691 | 100.91% | 4.87% | 7.83x | 稳定低 KV 高速点 |
| v417 expanded frontier 030 | 0.3673 | 100.40% | 4.95% | 8.11x | 当前最快完整 M100 点 |
| v415 expanded frontier 045 | 0.3714 | 101.51% | 5.05% | 7.82x | 质量略高，速度仍强 |
| v421 frontier-mode router | 0.3776 | 103.22% | 6.05% | 5.06x | 当前高分低 KV 点 |
| v419 macro-mode router | 0.3699 | 101.11% | 5.55% | 5.10x | 被其他点基本支配 |
| v396 old exact 5% | 0.3752 | 102.57% | 5.00% | 7.01x | 旧主线，仍然强 |

## 新候选

| 方法 | M20 Score | M20 KV | M20 Speed | 状态 |
|---|---:|---:|---:|---|
| v426 naive selective overlay | 0.3835 | 4.40% | 8.18x | M100 running |
| v427 source-preserving winners | 0.3989 | 4.71% | 8.32x | M100 running |
| v428 v427 + repobench | 0.4027 | 5.45% | 7.80x | M100 running |
| v424 latency-aware router | 0.3976 | 5.80% | 6.70x | M100 running |

## 当前判断

如果只看已经完成的 M100，v413/v417/v415/v396 构成低 KV 高速 Pareto：

- 最快：v417，4.95% KV，8.11x，100.40% full。
- 更稳质量：v415，5.05% KV，7.82x，101.51% full。
- 老强基线：v396，5.00% KV，7.01x，102.57% full。
- 高分 router：v421，6.05% KV，5.06x，103.22% full。

如果 v427/v428 M100 兑现 M20 趋势，它们会成为新的主线，因为它们同时具备：

- source-preserving 的方法故事；
- 约 5% KV；
- 约 8x speed；
- 明显高于 full baseline 的质量。

## 方法层面的新认识

naive overlay 会破坏 learned router 的 reference/fallback 语义。v426 的 M20 没有达到理论预期，正是因为 reference 从 v421 的 base 变成了 v417 的 base。

v427/v428 使用 `__task_sources` 保留完整 task fragment，因此 router 的 action policy、fallback、reference base 一起迁移。这比简单 task table 或简单 router overlay 更像一个可发表的方法点。

建议论文方法命名：`Source-Preserving Frontier Routing`。
