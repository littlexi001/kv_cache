# section184：Low-KV overnight 继续优化记录（v393-v397）

时间：2026-07-12 05:20 左右。

## 当前目标

目标不是继续盲扫参数，而是围绕一个清晰现象做方法推进：

- 不同 LongBench 任务的 KV 需求高度不均匀。
- `gov_report / multi_news / passage_retrieval / trec / triviaqa` 这类任务可以压到 1%-6% KV。
- `narrativeqa / hotpotqa / repobench-p` 是主要质量瓶颈，需要稀疏地保留更高预算。
- 因此合理方向是：先形成 5% / 7.5% / 10% 的 Pareto anchors，再训练 sample-level risk router 按样本风险升级预算。

## 已确认结果

目前已经完整 M100 确认的最好候选：

| 方法 | M100 score | KV ratio | speed vs full online | 备注 |
|---|---:|---:|---:|---|
| v385 quality10 | 0.3902 | 10.65% | 4.67x | 质量最高，但略超 10% KV |
| v389 task-knapsack-v2 | 0.3906 | 9.93% | 4.36x | 当前严格 10% 内最高完整 M100 |
| v386 task-knapsack-v378 | 0.3869 | 9.90% | 4.17x | v389 前的严格 10% 强基线 |
| v391 task-gated winner router | 0.3846 | 8.23% | 4.86x | 低于 v389，但给出更低 KV 的有效 Pareto 点 |
| v377 global pareto knapsack | 0.3811 | 9.78% | 4.52x | 严格 10% 内最稳的已完成 M100 |
| v387 planner-base-v377 | 0.3745 | 6.87% | 4.40x | 低 KV 稳定点，接近 v376 |
| v381 low-KV router | 0.3662 | 5.75% | 6.00x | 极低 KV，仍约等于 full baseline |

full baseline score 约为 `0.3658`，所以这些低 KV 方法在当前评测协议下已经达到或超过 full baseline。

## 新的现象驱动设计

离线用已完成 M100 候选做 task-level knapsack，得到一条可解释 Pareto 曲线：

| 目标 KV | 离线估计 score | 离线估计 KV | 离线估计 speed vs full |
|---:|---:|---:|---:|
| 5.0% | 0.3752 | 5.00% | 7.34x |
| 7.5% | 0.3849 | 7.49% | 5.15x |
| 10.0% | 0.3911 | 9.99% | 4.72x |

这说明后续论文故事可以从“固定预算压缩”转成“risk-aware Pareto action routing”：大多数样本用极低预算，少数高风险样本升级到高预算动作。

## 今晚新增后台实验

| run | 设计 | 当前状态 |
|---|---|---|
| v393 | 基于 v385 的严格 10% task-source composition | M20 已通过：score `0.4162`，KV `8.96%`，speed `5.15x`；M100 已自动启动 |
| v394 | exact 10% task-level knapsack | M20 running，通过后自动 M100 |
| v395 | exact 7.5% task-level knapsack | M20 已通过：score `0.4121`，KV `7.63%`，speed `3.78x`；M100 已自动启动 |
| v396 | exact 5% task-level knapsack | M20 已通过：score `0.4007`，KV `5.13%`，speed `6.32x`；M100 已自动启动 |
| v397 | 等 v393-v396 M100 后训练 cost-aware sample router，目标 KV<=10% | watcher 已启动，等待 Pareto anchors 完成 |
| v398 | v397 的 7.5% KV 目标版本 | watcher 已启动，等待 Pareto anchors 完成 |
| v399 | v397 的 5.5% KV 目标版本 | watcher 已启动，等待 Pareto anchors 完成 |
| v400 | 直接用当前已完成 M100 候选训练 cost-aware router | M20 已通过：score `0.3763`，KV `5.82%`，speed `6.18x`；M100 已自动启动 |
| v401 | v400 的 5.5% KV 目标版本 | 已启动训练，使用当前已完成 M100 候选 |

v397-v399 的作用：把 task-level 的 Pareto anchors 蒸馏为 sample-level router。它会以低 KV anchor 为 base，对每个样本预测是否值得升级到更高质量动作，并用 calibration/test gate 控制全局 KV。三个版本分别对应 `<=10% / <=7.5% / <=5.5%`，用于验证“同一套风险路由机制是否能自然形成完整 Pareto 曲线”。

## 服务器日志位置

- `outputs/logs/watch_v393_m100_task_knapsack_v385_20260712.log`
- `outputs/logs/watch_v394_m100_task_knapsack10_exact_20260712.log`
- `outputs/logs/watch_v395_m100_task_knapsack075_exact_20260712.log`
- `outputs/logs/watch_v396_m100_task_knapsack05_exact_20260712.log`
- `outputs/logs/watch_v397_after_pareto_20260712.log`
- `outputs/logs/watch_v398_cost_router075_after_pareto_20260712.log`
- `outputs/logs/watch_v399_cost_router055_after_pareto_20260712.log`
- `outputs/logs/watch_v400_cost_router_completed_m100_20260712.log`
- `outputs/logs/watch_v401_cost_router055_completed_m100_20260712.log`

## 新完成的 M100 诊断

v389 是当前严格 10% 内最好完整 M100：score `0.3906`，KV `9.93%`，speed `4.36x`。它的任务级形态很清楚：

- `hotpotqa`: score `0.4604`，KV `40.52%`。
- `repobench-p`: score `0.5008`，KV `39.15%`。
- `2wikimqa`: score `0.3342`，KV `14.64%`。
- `qasper`: score `0.2905`，KV `11.33%`。
- 大量其它任务保持在 `1%-7%` KV。

这说明 v389 的优势来自“把少数高风险任务预算打开，同时极端压缩容易任务”，而不是全局均匀保留 KV。

v386 相对 v377 的主要提升来自：

- `hotpotqa`: `+0.0871`，但 KV 达到 `40.52%`，说明 HotpotQA 是高风险任务，需要局部高预算。
- `2wikimqa`: `+0.0302`，KV `16.53%`。
- `musique`: `+0.0165`，KV `9.25%`。
- `triviaqa`: `+0.0139`，KV `2.27%`。

v386 的主要退化来自：

- `narrativeqa`: `-0.0313`，说明 narrativeqa 需要更接近 v385 的高质量动作，而不是 v386 的低预算版本。
- `repobench-p / qmsum / lcc` 有小幅退化。

这个结果支持下一步 sample-level router：不是所有任务都该统一提高预算，而是只在 HotpotQA、NarrativeQA、RepoBench 这类高风险样本上升级动作。

## v400：即时 sample-level router 现象

v400 不等待 v393-v396 的新 anchors，而是直接使用当前已完成 M100 候选：

- base action: `policy_v381`，这是一个约 `5.75%` KV 的低预算强基线。
- selected tasks: `2wikimqa, hotpotqa, lcc, multifieldqa_en, narrativeqa, qasper, qmsum, repobench-p, samsum, triviaqa`。
- offline all: score `0.3966`，KV `9.34%`，speed `5.15x`。
- calibration gain `+0.0140`，test gain `+0.0177`，说明不是明显训练集记忆。
- actual M20: score `0.3763`，KV `5.82%`，speed `6.18x`，已通过 M100 gate。

最重要的特征包括 `raw_prompt_tokens / page_count / context_length / score gap / query coverage`，这符合“根据检索稳定性和样本风险升级预算”的论文叙事。

注意：v400 的实跑 KV 明显低于 offline 估计，因此它目前更像一个 `5%-6% KV` 的强低预算点，而不是 `10% KV` 质量优先点。明早需要重点看它的 M100 是否还能稳定超过 full baseline。

v392 的 winner-router 离线被 gate 拦下：all score `0.4120` 很高，但 test gain `-0.0049`，说明它可能过拟合，因此没有进入 M20/M100。这个结果反过来支持 v400 这种 cost-aware、calibration/test 双 gate 的路由方式。

自动进度汇总：

- `outputs/riskkv_lowkv_running_progress_20260712.json`
- `doc/section172_lowkv_running_progress_20260712.md`

## 明早优先看什么

1. 先看 v393/v394/v395/v396 的 M100 是否完成，以及三条 Pareto anchors 是否都超过 full baseline。
2. 如果 v397 自动训练并通过 M20/M100，优先比较 v397 与 v394/v395/v396：v397 如果能在接近 5%-8% KV 下接近 10% anchor 的质量，就是更像 ICLR 主方法的结果。
3. 如果 v397 不提升，论文主线仍可退回到 task-level Pareto planner：它已经给出了强、可解释、可复现的 KV-quality-speed 曲线。
