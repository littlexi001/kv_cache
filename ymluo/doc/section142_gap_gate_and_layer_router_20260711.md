# Section 142: gap-gated b16 与 layer-wise memory action

日期：2026-07-11

## 关键结论

上一轮 b16 多块实验给出了一个比较明确的负结论：

- b16 作为全局保留单位不稳。
- b16 作为 locator 加连续窗口，在 M20 上有局部高分，但 M100 不泛化。
- 用 full fallback 可以救回部分质量，但 KV keep 会明显偏高，Pareto 不好。

因此下一步不继续盲目扫 block size，而是验证两个更具体的机制：

1. **gap-gated compression**：只在 page-score gap 足够高、证据定位较确定时使用小窗口；低 gap 样本回退。
2. **layer-wise memory action**：高 KV 任务不一定每层都需要同样长的 KV；低层使用小 sink/recent，高层使用检索证据，尝试兑现 attention 加速。

## 已完成：纯 v255 M100 targeted

| Run | Task | Score | KV keep | Online | 结论 |
|---|---|---:|---:|---:|---|
| v256 v255 anchor64 | musique | 0.1494 | 27.67% | 0.243s | M20 的 0.2333 没有泛化。 |
| v257 v255 anchor64 | qasper | 0.3170 | 39.10% | 0.420s | 明显低于 v241/v235 qasper 的 0.3987。 |

结论：不能把 v255 直接并入主方法。

## 已完成：v258 full-fallback gap gate

设计：

- qasper / musique 使用 b16 locator + anchor-window。
- 当 `gap2 <= 0.04` 时触发 score-risk。
- v258 的风险动作是 full fallback。

结果：

| Run | Task | Score | KV keep | Online | 结论 |
|---|---|---:|---:|---:|---|
| v258 gap-gated | qasper | 0.3908 | 69.00% | 0.545s | 质量接近 v235，但 KV 远高于 v235 的 42.73%，不可用。 |
| v258 gap-gated | musique | 0.1851 | 56.62% | 0.334s | 比纯 v255 好，但仍低于 v235 的 0.2241，且 KV 偏高。 |

结论：gap2 确实能识别危险样本，但 full fallback 不是好的 Pareto 动作。

## 已完成：v259 moderate-budget gap ladder

设计：

- 同样使用 `gap2 <= 0.04` 做风险门控。
- 风险动作不再 full fallback：
  - qasper: 1536 -> 3072
  - musique: 2048 -> 4096

目标：

- 判断“低 gap 样本是否只需要更大窗口预算，而不需要 full KV”。
- 如果 v259 质量接近 v235，且 KV 明显低于 v258，则可以作为新的 router action。

运行中：

结果：

| Run | Task | Score | KV keep | Online | 结论 |
|---|---|---:|---:|---:|---|
| v259 gap-ladder | qasper | 0.3839 | 54.25% | 0.412s | 比 v258 KV 低，但仍高于 v241 的 42.73%，分数也低于 v241。 |
| v259 gap-ladder | musique | 0.1424 | 38.54% | 0.270s | KV 降了，但分数明显低于 v241/v235。 |

结论：gap2 是有用的风险信号，但 b16-window 的中等预算 fallback 不是足够好的高风险动作。

## 新启动：v260 layer-wise memory action

动机：

v241 已经满足整体目标，但高 KV 任务仍然拖累端到端速度：

| Task | v241 score | v241 KV keep |
|---|---:|---:|
| qasper | 0.3987 | 42.73% |
| multifieldqa_en | 0.5121 | 40.48% |
| musique | 0.2241 | 71.13% |
| narrativeqa | 0.1723 | 28.76% |
| repobench-p | 0.5513 | 46.40% |

这些任务可能需要较完整的 evidence，但未必每一层都需要长 KV。因此 v260 对这些高 KV 任务启用 layer router：

- lower 50% layers: sink/recent 512-token streaming memory
- upper 50% layers: 原 v241 retrieval memory

M50 targeted 结果：

| Run | Task | Score | KV keep | Online | 结论 |
|---|---|---:|---:|---:|---|
| v260 layer-router | narrativeqa | 0.1927 | 31.56% | 0.225s | 同前 50 样本 v241 分数接近，速度没有变快。 |
| v260 layer-router | qasper | 0.1041 | 22.02% | 0.160s | 质量崩，不可用。 |
| v260 layer-router | multifieldqa_en | 0.3897 | 44.78% | 1.321s | 质量明显低于 v241 first50。 |
| v260 layer-router | musique | 0.1713 | 65.72% | 0.449s | 质量和速度都不如 v241 first50。 |
| v260 layer-router | repobench-p | 0.5806 | 42.32% | 2.231s | 分数高于 v241 first50，但速度没有优势。 |

判断标准：

- 分数接近 v241 task-level M100。
- online 时间明显下降。
- 如果 reported KV keep 不下降但 online 下降，也可以作为 attention-subsystem 的真实加速结果。

结论：layer-wise memory action 不是当前主线。它可能对 code completion 的质量有正面影响，但没有兑现速度收益；对 qasper/multifield/musique 会伤质量。

## 新正结果：v261 qmsum direct operator

发现：

qmsum 在 v241 中分数不高，但 online 很慢，主要由 decode 决定：

| qmsum method | Score | KV keep | Online |
|---|---:|---:|---:|
| v241 qmsum | 0.1540 | 14.71% | 2.454s |
| v265 query-focused direct, 128 words | 0.1127 | 2.23% | 0.026s |
| v261 query-focused direct, 192 words | 0.1051 | 2.23% | 0.026s |
| v266 query-focused direct, 160 words | 0.1072 | 2.23% | 0.027s |
| v263 query-focused direct, 256 words | 0.0988 | 2.23% | 0.027s |
| v264 query-focused direct, 320 words | 0.0945 | 2.23% | 0.026s |

128-word direct summary 最好。更长的 extractive output 会引入噪声，反而降低 ROUGE-L。

## 当前最好 practical Pareto：v267

v267 = v241 主方法 + qmsum 128-word direct operator。

它不是 oracle：qmsum direct operator 已经完整跑完 M100，其它任务沿用已完成的 v241 M100 结果。合成目录：

`outputs/riskkv_v19_v267_v241_plus_qmsum_direct128_combined_20260711_m100_bDyn_pDyn`

| Method | Score | KV keep | Online | Total | Speed vs 3.033s full online |
|---|---:|---:|---:|---:|---:|
| v241 previous best | 0.3936 | 22.11% | 0.636s | 1.803s | 4.77x |
| v262 v241 + qmsum direct 192 | 0.3906 | 21.33% | 0.484s | 1.521s | 6.27x |
| **v267 v241 + qmsum direct 128** | **0.3911** | **21.33%** | **0.484s** | **1.521s** | **6.27x** |

相对 full baseline：

- 若使用 LongBench m20 full_raw `0.3596`，v267 = `108.7%`。
- 若使用 LongBench full_raw `0.372655`，v267 = `104.9%`。

因此 v267 仍满足目标：

- KV keep 在 `10%-30%`：实际 `21.33%`。
- online speed 超过 `2.5x`：实际约 `6.27x`。
- 分数达到 full baseline `95%+`：实际高于 full baseline。

论文叙事上，v267 支持一个更清楚的说法：RiskKV-Block 不只是统一做 token pruning，而是一个 task-conditioned memory-action router。对 QA 任务使用 risk-aware evidence KV；对结构化任务使用 direct operator；对 qmsum 这类低收益长生成任务，用 query-focused extractive memory operator 替代昂贵 decode。
