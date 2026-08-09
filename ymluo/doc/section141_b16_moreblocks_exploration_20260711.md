# Section 141: b16 多块选择继续探索

日期：2026-07-11

## 背景

用户提出一个合理质疑：`block_size=16` 不一定太碎，因为 token 本身就是逐 token 处理；如果 16-token block 选得更多，可能比 128-token block 更细粒度、更高召回。

因此这轮实验不把 b16 直接判死，而是拆成两个问题：

1. b16 选更多块后，LongBench QA 质量能否恢复。
2. 如果质量恢复，KV keep 和 online speed 是否仍在可发表的 practical 区间。

## 已完成负结果

### graph-bridge targeted M100

| Run | Task | Score | KV keep | Online | 结论 |
|---|---:|---:|---:|---:|---|
| v250 qasper graph-bridge risk1536 | qasper | 0.3783 | 34.03% | 0.380s | 低于 v235 qasper path 的 0.3987，不并入主线。 |
| v251 musique graph-bridge risk3072 | musique | 0.1351 | 32.92% | 0.221s | 明显低于 v235 musique path 的 0.2241。 |
| v252 musique graph-bridge risk4096 | musique | 0.1358 | 41.76% | 0.226s | 增加预算没有恢复质量。 |
| v245 musique b16 all-block risk3072 | musique | 0.1250 | 33.10% | 0.268s | “b16 多选 + capped risk”不能解决 musique。 |

结论：musique/qasper 的主要瓶颈不是简单 block 更细或预算略大，而是需要更可靠的 multi-hop evidence composition。

## 新启动实验

这轮启动三组 M20 全任务探针，目标是直接回答“b16 多选是否可行”。

| Run | Policy | 设计 | 目标 |
|---|---|---|---|
| v253 | `riskkv_task_policy_v253_b16_moreblocks_balanced_20260711.json` | b16 all-block；QA 预算中等放大；关闭 full score-risk fallback | 测 10%-30% KV 附近是否能恢复质量。 |
| v254 | `riskkv_task_policy_v254_b16_moreblocks_highrecall_20260711.json` | b16 all-block；QA 预算进一步放大 | 测 b16 多选的质量上界。 |
| v255 | `riskkv_task_policy_v255_b16_moreblocks_anchor64_20260711.json` | b16 定位，但保留 64/96-token 连续窗口 | 验证 16-token 证据是否需要局部连续上下文。 |

当前判断规则：

- 如果 v254 质量仍低于 v241/v235，说明“多选 16-block”不是主要方向。
- 如果 v254 质量高但 KV 太高，下一步训练 sample-level router，只在危险样本上放大 b16 预算。
- 如果 v255 明显优于 v253，说明 b16 更适合作为 locator，而不是最终 KV 保留单位。
- 如果 v253 优于 v255，说明用户的判断成立：16-token block 本身不碎，核心是选块数量和排序。

## 已完成结果：v255

`v255_b16_moreblocks_anchor64_m20` 已完成 M20 全任务：

| Method | Samples | Score | KV keep | Online | Total |
|---|---:|---:|---:|---:|---:|
| v255 b16 locator + 64/96-token anchor window | 320 | 0.3898 | 17.97% | 0.596s | 1.766s |

任务级结果：

| Task | Score | KV keep | Online |
|---|---:|---:|---:|
| narrativeqa | 0.1158 | 10.87% | 0.184s |
| qasper | 0.4440 | 35.68% | 0.532s |
| multifieldqa_en | 0.3924 | 16.26% | 1.260s |
| hotpotqa | 0.2602 | 31.27% | 0.150s |
| 2wikimqa | 0.3787 | 60.30% | 0.386s |
| musique | 0.2333 | 27.63% | 0.262s |
| gov_report | 0.1807 | 2.36% | 0.005s |
| qmsum | 0.1564 | 14.57% | 2.079s |
| multi_news | 0.1734 | 10.72% | 0.001s |
| trec | 0.8000 | 2.82% | 0.227s |
| triviaqa | 0.2062 | 9.81% | 1.193s |
| samsum | 0.2577 | 5.64% | 1.053s |
| passage_count | 0.4000 | 2.46% | 0.010s |
| passage_retrieval_en | 1.0000 | 2.06% | 0.012s |
| lcc | 0.6607 | 15.84% | 0.534s |
| repobench-p | 0.5767 | 39.27% | 1.648s |

初步结论：

- v255 比 v236 anchor-window capped-risk 更好一些，但没有超过当前 v241 M20 的 0.4035。
- b16 locator + 连续窗口在 qasper/musique 有局部积极信号；其中 musique M20 达到 0.2333 / 27.63% KV。
- 2wikimqa 和 repobench-p 的 KV 明显偏高，说明窗口策略需要 task-level 或 sample-level router，而不能全局使用。
- 当前不能把 v255 替换为主方法，但它支持一个论文故事：小 block 更适合作为 evidence locator，最终保留单位应由任务动态决定。

## 已完成结果：v253 / v254

| Run | Samples | Score | KV keep | Online | 结论 |
|---|---:|---:|---:|---:|---|
| v253 b16 moreblocks balanced M20 | 320 | 0.3804 | 17.75% | 0.587s | KV 好，但分数低于 v241/v255。 |
| v254 b16 moreblocks high-recall M20 | 320 | 0.3908 | 21.87% | 0.593s | 高召回预算恢复了一些分数，但仍不优于 v241/v262。 |

## 已完成结果：v256 / v257 M100

| Run | Task | Samples | Score | KV keep | Online | 结论 |
|---|---|---:|---:|---:|---:|---|
| v256 v255 musique anchor64 | musique | 100 | 0.1494 | 27.67% | 0.243s | M20 的 0.2333 没有泛化。 |
| v257 v255 qasper anchor64 | qasper | 100 | 0.3170 | 39.10% | 0.420s | 明显低于 v241/v235 qasper。 |

最终结论：b16 可以作为局部 locator，但“b16 多选/窗口”不是当前最好的主方法。真正有价值的新方向转向 Section 142：qmsum direct operator 带来了更好的全局 speed-quality Pareto。
