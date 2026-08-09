# Section 187: v427 Selective Source Router 结果与判断

更新时间：2026-07-12

Full baseline：LongBench M100，score=0.3658，online=3.0988s。

## 当前最重要结果

已完成 M100：

| 方法 | Score | vs full | KV | Speed | 判断 |
|---|---:|---:|---:|---:|---|
| v417 expanded frontier 030 | 0.3673 | 100.40% | 4.95% | 8.11x | 当前最快且已完整验证的低 KV 点 |
| v415 expanded frontier 045 | 0.3714 | 101.51% | 5.05% | 7.82x | 比 v417 高分，速度略低 |
| v421 frontier-mode router | 0.3776 | 103.22% | 6.05% | 5.06x | 当前最高分低 KV M100 点之一 |
| v419 macro-mode router | 0.3699 | 101.11% | 5.55% | 5.10x | 被 v417/v415 支配 |
| v398 cost router 7.5% | 0.3776 | 103.22% | 7.04% | 4.71x | 被 v421 支配 |

新完成/运行中的 M20/M100：

| 方法 | M20 Score | M20 KV | M20 Speed | 状态 |
|---|---:|---:|---:|---|
| v413 expanded frontier 035 | 0.3879 | 4.62% | 8.61x | M100 running |
| v424 latency-aware frontier router | 0.3976 | 5.80% | 6.70x | M100 queued/running |
| v426 selective overlay | 0.3835 | 4.40% | 8.18x | M100 running |
| v427 selective source router | 0.3989 | 4.71% | 8.32x | M100 running |

## 关键发现

v421 的 M20 很强，但全局打开 router 后速度掉到 5-6x。逐任务分析显示：

- v421 明显优于 v417 的任务：`narrativeqa`, `qasper`, `musique`, `multifieldqa_en`, `lcc`。
- v421 不适合覆盖的任务：`hotpotqa`, `2wikimqa`, `multi_news`, `samsum`, `repobench-p`。这些任务要么掉分，要么明显拖慢。

第一次尝试 v426 只把 v421 的 router 参数 overlay 到 v417 上，M20 只到 0.3835。原因是 v421 的 `reference` 动作原本回退到 v396，而 v426 overlay 后 `reference` 变成回退到 v417，破坏了 router 的语义。

v427 改成对获胜任务直接 `__task_sources` 到 v421 的完整 task fragment，保留 v421 的 reference/fallback 语义；其他任务仍保留 v417 的高速路径。结果 M20 达到：

- score=0.3989
- KV=4.71%
- speed=8.32x

这是目前最强的候选主线。如果 M100 兑现，v427 会优于 v417/v415/v421 的单独版本。

## 方法故事更新

现在论文主线可以更清晰：

1. 先构建多个低 KV frontier mode。
2. 用 M20/M100 发现不同任务/样本的胜出 mode 不同。
3. 不是全局套用 router，而是做 calibrated selective source routing：
   - fast base 保证低 KV 和高速度；
   - winner task source 保留完整 fallback 语义；
   - runtime router 只在被验证有收益的任务片段内生效。

这比“简单 task table”更有论文价值，因为核心发现是 reference/fallback 语义不能被 naive overlay 破坏，必须把 action policy 和 base frontier 作为一个完整 source 迁移。

## 下一步

1. 等 v427 M100 完成，优先比较 v427 / v421 / v415 / v417。
2. 如果 v427 M100 仍保持 M20 趋势，主方法设为 v427。
3. 补 ablation：
   - v417 fast base。
   - v421 full router。
   - v426 naive overlay。
   - v427 source-preserving selective router。
4. 把论文中的方法命名从单纯 router 调整为 `Source-Preserving Frontier Routing`。
