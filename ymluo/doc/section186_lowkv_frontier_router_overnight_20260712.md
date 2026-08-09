# Section 186: Low-KV Frontier Router 夜间优化进展

更新时间：2026-07-12

目标：LongBench 上 KV ratio 保持在 1%-10%，端到端 online speed 达到 2.5x+，分数达到 full baseline 的 95%+。Full baseline 使用同一批 M100 样本，score=0.3658，online=3.0988s。

## 当前结论

1. 目前最稳的已完成 M100 主线仍是 v396：score=0.3752，KV=5.00%，speed=7.01x，约为 full 的 102.57%。
2. 新完成的 v404/v405 能略降 KV，但没有实质超过 v396：
   - v404：score=0.3746，KV=4.99%，speed=6.35x，约为 full 的 102.40%。
   - v405：score=0.3719，KV=4.79%，speed=6.37x，约为 full 的 101.67%。
3. v398 分数更高，但成本也更高：score=0.3776，KV=7.04%，speed=4.71x，约为 full 的 103.22%。它适合作为 Pareto 曲线上的高质量点，不适合作为极低 KV 主线。
4. 今晚最重要的新现象来自 expanded frontier 和 frontier router：
   - v417 M20：score=0.3738，KV=4.90%，speed=7.77x。
   - v421 M20：score=0.4008，KV=4.82%，speed=5.77x。
   - v415 M20：score=0.3965，KV=5.29%，speed=7.55x。
5. v417/v421/v415 的 M100 已经排队或运行中，其中 v417 和 v421 已开始跑 M100，v415 仍在等空闲 GPU。

## 新方法：Frontier-Mode Router

这次不是继续盲目扫固定预算，而是把多个已知有效的策略视为可选择的“frontier mode”：

- frontier_030 / 035 / 040 / 045 / 050 / 075 / 100
- legacy_b128 / legacy_b256 / legacy_b512
- operator_direct

训练时使用 v396 的稳定 runtime features，包括：

- prompt/context length
- page count
- block score 的 max/mean/gap/entropy/positive fraction
- query coverage recall
- task family，但不使用 task one-hot

router 学习在不同 frontier mode 之间切换。默认 reference 是 v396，因此低置信度时退回已验证的 5% 强基线。

离线 proxy 结果：

| 方法 | Split | Score | vs full | KV | Speed |
|---|---|---:|---:|---:|---:|
| v421 frontier router | calibration | 0.3390 | 95.87% | 4.22% | 6.35x |
| v421 frontier router | test | 0.3822 | 105.57% | 4.13% | 7.81x |
| v421 frontier router | all proxy | 0.3936 | 107.61% | 4.91% | 5.66x |

真实 M20 结果：

| 方法 | Score | vs full M100 | KV | Speed | 状态 |
|---|---:|---:|---:|---:|---|
| v417 expanded frontier 030 | 0.3738 | 102.20% | 4.90% | 7.77x | M100 running |
| v421 frontier router 035 | 0.4008 | 109.56% | 4.82% | 5.77x | M100 running |
| v415 expanded frontier 045 | 0.3965 | 108.38% | 5.29% | 7.55x | M100 queued |

说明：v417/v421/v415 都因为原始 gate 使用了名义 KV 上限而被误拦过。例如 v417 标称 3%，实际 M20 是 4.90%；v421 标称 3.5%，实际 M20 是 4.82%。这些实际结果仍然完全满足论文目标的 1%-10% 范围，因此已用 relaxed gate 补排 M100。

## 已完成 M100 对照

| 方法 | Score | vs full | KV | Speed | 判断 |
|---|---:|---:|---:|---:|---|
| v396 exact 5% task knapsack | 0.3752 | 102.57% | 5.00% | 7.01x | 当前最稳主线 |
| v404 pair planner after-pareto 10% | 0.3746 | 102.40% | 4.99% | 6.35x | 接近 v396，但不明显更好 |
| v405 pair planner after-pareto 7.5% | 0.3719 | 101.67% | 4.79% | 6.37x | 更省 KV，但掉分 |
| v406 pair planner base381 | 0.3727 | 101.89% | 5.69% | 5.09x | 不作为主线 |
| v398 cost router 7.5% | 0.3776 | 103.22% | 7.04% | 4.71x | 高质量 Pareto 点 |
| v401 cost router 5.5% | 0.3719 | 101.67% | 7.47% | 5.34x | 不如 v396 |
| v402 task-gated pair planner 10% | 0.3719 | 101.66% | 5.66% | 4.77x | 不如 v396 |
| v403 task-gated pair planner 7.5% | 0.3639 | 99.47% | 5.11% | 5.19x | 不如 v396 |

## 关键发现

1. 3% 以下的 proxy 很诱人，但真实运行会因为 policy 继承、fallback、recent/sink 和任务级特殊路径把 KV 抬到约 4.8%-5.3%。这不是坏事，因为质量显著更稳，而且仍在目标压缩率内。
2. 困难任务主要是 2Wiki、HotpotQA、Musique、Qasper、NarrativeQA；多数 synthetic/few-shot/summary 任务可以在 1%-6% KV 内保持质量。
3. 只做低预算固定策略不够。更有希望的是“低 KV frontier + 风险感知 router + 保守 reference fallback”。
4. v421 没有使用 task one-hot，只用 family 和 runtime 检索置信特征，比纯任务表更适合作为论文故事。

## 当前后台任务

- v417 M100：running。
- v421 M100：running。
- v415 M100：queued。
- 其他旧分支已完成或接近完成，但目前没有超过 v396/v417/v421 主线的迹象。

## 下一步

1. 等 v417/v421/v415 M100 完成后，优先比较这三个点和 v396：
   - 如果 v421 M100 保持 M20 趋势，主方法改成 Frontier-Mode Router。
   - 如果 v421 M100 波动大但 v417/v415 稳，主方法用 expanded frontier，router 作为扩展实验。
2. 对 v421 做 ablation：
   - 不使用 task one-hot。
   - 去掉 legacy sparse candidates。
   - 去掉 operator_direct。
   - 只用 task-level frontier，不用 sample-level router。
3. 论文故事建议聚焦为：从 fixed budget KV compression 转向 risk-calibrated frontier routing，在低 KV 区间自动选择最小安全 frontier。
