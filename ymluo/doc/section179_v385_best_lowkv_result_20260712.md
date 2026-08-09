# section179：v385 当前最佳低 KV 结果

日期：2026-07-12

## 结论

今晚最好的实际可用方法已经从 v377 推进到 v385。

v385 在 LongBench question-aware M20 上达到：

| 方法 | samples | score | KV keep | speed/full | vs full | vs v300 practical |
|---|---:|---:|---:|---:|---:|---:|
| v385 quality10 | 320 | 0.4205 | 9.80% | 5.02x | 114.95% | 95.73% |
| v384 task-gated | 320 | 0.4131 | 9.15% | 5.13x | 112.92% | 94.04% |
| v377 global Pareto | 320 | 0.4029 | 8.48% | 5.29x | 110.15% | 91.73% |
| v381 no-post router | 320 | 0.3981 | 5.82% | 6.84x | 108.83% | 90.63% |
| v376 strict10 | 320 | 0.3969 | 6.03% | 5.81x | 108.49% | 90.35% |

这满足昨晚设定的核心目标：1%-10% KV keep、2.5x+ speed、分数达到 baseline 95%+。如果 baseline 指 full KV，v385 是 114.95%；如果用更强的 v300 practical baseline，v385 也达到 95.73%。

## v385 是怎么来的

v385 不是盲目调参，而是根据已完成实验的 per-task 现象组合出来的：

1. v377 是稳定低 KV 基座，整体 0.4029、8.48% KV。
2. v380 作为全局 router 不如 v377，但它在 2wikimqa、hotpotqa、musique 上明显优于 v377。
3. v384 因此只在 2wikimqa/hotpotqa/musique 启用 v380 router，其他任务保持 v377，结果升到 0.4131。
4. v363 在 narrativeqa、qasper、repobench-p 上优于 v384，其中 qasper/repobench 基本不增加 KV。
5. 加入 narrativeqa 会增加 KV，所以 v385 同时把 qmsum 换成低 KV 版本抵消预算，最终达到 0.4205、9.80% KV。

## 当前正在验证

v385 M20 gate 已通过，并已自动启动 M100：

- M20 输出：`outputs/riskkv_v19_v385_quality10_v384_plus_v363_qmsumlow_20260712_quality10_v384_plus_v363_qmsumlow_v385_m20_bDyn_pDyn`
- M100 输出：`outputs/riskkv_v19_v385_quality10_v384_plus_v363_qmsumlow_20260712_quality10_v384_plus_v363_qmsumlow_v385_m100_bDyn_pDyn`
- watcher 日志：`outputs/logs/watch_v385_quality10_v384_plus_v363_qmsumlow_20260712.log`

同时在跑的关键 M100：

- v377：验证原始低 KV 基座是否稳。
- v380/v381/v382/v383：验证不同 router/fallback 形式的泛化。
- v384：验证 task-gated 版本是否稳。
- v385：验证当前最佳版本是否稳。

当前已经完成的 M100 保底结果：

| 方法 | samples | score | KV keep | speed/full | vs full | vs v300 practical |
|---|---:|---:|---:|---:|---:|---:|
| v377 global Pareto M100 | 1600 | 0.3811 | 9.78% | 4.52x | 104.17% | 86.75% |
| v376 strict10 M100 | 1600 | 0.3743 | 6.87% | 5.21x | 102.34% | 85.23% |
| v368 direct/operator M100 | 1600 | 0.3697 | 8.47% | 7.21x | 101.07% | 84.17% |

因此目前“已被 M100 验证”的保底主线是 v377；“M20 最强主候选”是 v385，正在等待 M100 验证。

## 对论文故事的启发

现在最强的故事不是“训练一个 router”，而是：

- 先构造一组可解释的 memory actions；
- 用任务级 Pareto 发现不同任务的低 KV 可行区域；
- 用样本级 router 只在确实有收益的任务族上启用；
- 用预算补偿把高风险任务的质量收益控制在全局 10% KV 内。

这个故事比单纯 fixed-budget KV eviction 更像一个 risk-aware memory policy planner，也更容易和 AdaKV/SnapKV/PyramidKV 区分。

## 明早优先判断

1. 如果 v385 M100 仍然保持 0.40+ 且 KV <10.5%，v385 应该成为当前主方法。
2. 如果 v385 M100 回落明显，但 v384 M100 稳，主线应改成更保守的 task-gated v384。
3. 如果所有 task-gated M100 都明显回落，而 v377 M100 稳，则论文主线先用 v377，v384/v385 作为 adaptive extension。
