# section173：低 KV 全局 Pareto v377 设计与夜间实验记录

日期：2026-07-12

目标不是继续盲扫预算，而是利用已经观察到的任务级现象，把全局平均 KV 控制在 1%-10%，同时尽量提高 LongBench 质量和端到端速度。

## 已确认的 M20 现象

| 方法 | score | KV keep | speed/full | vs full | vs v300 |
|---|---:|---:|---:|---:|---:|
| v365 ultra skeleton | 0.3776 | 8.24% | 5.79x | 103.23% | 85.97% |
| v368 direct operator extreme mix | 0.3754 | 7.99% | 8.18x | 102.64% | 85.48% |
| v363 taskwise lowkv mix | 0.4096 | 14.17% | 4.75x | 111.98% | 93.26% |
| v360 lowkv certificate | 0.3948 | 10.05% | 5.39x | 107.92% | 89.88% |
| v373 selective direct ladder | 0.3070 | 5.12% | 14.21x | 83.91% | 69.88% |

结论：

- 相对 full KV baseline，v365/v368 已经满足 1%-10% KV、2.5x+ speed、95%+ score 的目标。
- 相对当前 practical baseline v300，v365/v368 还不够，主要差距来自 narrativeqa/qasper/hotpotqa/repobench-p 这些证据闭包任务。
- direct QA/extractive QA 不能作为主线：v373 速度很好，但 qasper/multifieldqa/triviaqa 等任务质量明显下降。

## 关键现象

全局平均 KV 约束比“每个任务都小于 10% KV”更合理。部分任务天然可以用 1%-6% KV 保分，例如 gov_report、multi_news、trec、passage retrieval/count、triviaqa、samsum、qmsum；这些任务可以抵消 narrativeqa、qasper、repobench-p 等少数高风险任务需要的更大预算。

基于已完成 M20 的非支配候选，做任务级 knapsack 估计后，理论组合可以达到：

| 约束 | 估计 score | 估计 KV keep | 估计 online |
|---|---:|---:|---:|
| avg KV <= 5% | 0.3903 | 4.92% | 0.4898s |
| avg KV <= 7% | 0.4091 | 6.93% | 0.5720s |
| avg KV <= 10% | 0.4156 | 9.31% | 0.6117s |

这个估计不是 oracle sample 级选择，而是把已验证的任务级 policy 组合起来；仍然需要 M20/M100 真实运行验证。

## v377 方案

新增配置：

`ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v377_global_pareto_knapsack_20260712.json`

v377 继承 v375，但做两个关键改动：

1. narrativeqa 使用 v363 风格的高质量风险升级：保留 v300 的 score-risk escalation，让少数高风险样本回到大预算或 full-like 行为。这样牺牲该任务 KV，但能显著修复 narrativeqa 分数。
2. hotpotqa 使用 v373/v376 风格的低 KV graph/laddder：用 192 起步的细粒度 block、coverage certificate、graph bridge 和 budget ladder，降低 hotpotqa 的平均 KV，抵消 narrativeqa 的高 KV。

直觉：把“需要证据闭包的任务”分成两类处理，而不是一刀切。narrativeqa 对证据定位不稳定，允许风险升级；hotpotqa 更适合 graph bridge + ladder，在低 KV 下接受一定质量损失来换全局预算。

## 自动实验

新增 watcher：

`ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/watch_v377_global_pareto_20260712.sh`

逻辑：

- 先跑 v377 M20。
- M20 gate：score >= 0.38，KV keep <= 10.5%，speed/full >= 2.5x。
- 只有 gate 通过才自动跑 M100。

服务器上已提交后台 watcher：

`outputs/logs/watch_v377_global_pareto_20260712.log`

提交后本机到 `10.176.37.31:22` 出现连接超时，因此后续状态还没有二次确认；恢复 SSH 后需要第一时间检查该 log、v377 M20 gate flag 和是否已经进入 M100。

如果 M100 通过，v377 会成为下一轮主方法候选；如果 v377 失败，结论也有价值：说明任务级 M20 Pareto 组合存在样本外不稳，需要转向 sample-level policy/router，而不是继续人工拼任务 policy。
