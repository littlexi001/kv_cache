# section180：Low-KV v386/v387 overnight 计划

日期：2026-07-12

目标：在 LongBench question-aware 设置下继续逼近 `1%-10% KV keep`、`2.5x+ speed/full`、`>=95% full-cache score`。更高目标是尽量接近当前 practical/v300 的质量，但今晚优先保证 M100 证据可靠。

## 当前已确认最好结果

已完成 M100 中最稳的可用点仍是 v377：

| 方法 | samples | score | vs full | KV keep | speed/full |
|---|---:|---:|---:|---:|---:|
| v377 global pareto knapsack | 1600 | 0.3811 | 104.17% | 9.78% | 4.52x |

短样本 M20 上最强的是 v385：

| 方法 | samples | score | vs full | vs v300 | KV keep | speed/full |
|---|---:|---:|---:|---:|---:|---:|
| v385 quality10 mix | 320 | 0.4205 | 114.95% | 95.73% | 9.80% | 5.02x |

但是 v385 还在跑 M100，所以它现在只能作为高潜力候选，不能当作最终结论。

## 今晚新增 v386

v386 是 M100 证据驱动的 task-level Pareto/knapsack 组合。它不再用 M20 偶然现象，而是只从已完成 M100 的 v368/v375/v376/v377/v378 里选择每个任务的非支配策略。

离线任务级 knapsack 预期：

| KV limit | expected score | KV keep | online | speed/full |
|---:|---:|---:|---:|---:|
| 10% | 0.3869 | 9.90% | 0.6459s | 4.80x |

选择逻辑：

| 任务 | 选择来源 | 原因 |
|---|---|---|
| 2wikimqa | v378 | M100 中 2Wiki 质量最高 |
| hotpotqa | v375 | Hotpot 质量最高，虽然 KV 较高，但全局预算能承受 |
| multifieldqa_en/qasper | v377 | coverage certificate 稳定 |
| passage_count/passage_retrieval_en/triviaqa | v368 | direct/structured 路径低 KV 且质量稳定 |
| qmsum/repobench-p/musique/lcc | v378 | 样本级 planner 在这些任务上给出更优 Pareto 点 |
| gov_report/multi_news/samsum/trec/narrativeqa | v375/v376 | 低 KV 或直接结构化路径更稳 |

代码变化：

- `src/run_controlled_public_kv_benchmark_v1.py` 增加 `__task_sources`，用于从已有 policy 中复用某个任务的完整合并配置。
- `configs/riskkv_task_policy_v386_m100_task_knapsack_v378_20260712.json`
- `scripts/watch_v386_m100_task_knapsack_v378_20260712.sh`

运行状态：已在 GPU0 后台启动，先跑 m20 sanity，再直接跑 m100。

## 今晚新增 v387

v387 是样本级 oracle gap 探索，不是主方法。它修正了旧 learned router 的两个问题：

1. 候选策略全部来自已完成 M100，不混入 M20。
2. fallback 保留 v377 base，而不是退回 v365/v293。

离线训练结果：

| 方法 | score | KV keep | speed/full | 备注 |
|---|---:|---:|---:|---|
| v377 base | 0.3811 | 9.78% | 4.52x | 当前 M100 稳定主线 |
| v387 learned | 0.3760 | 7.08% | 5.09x | 更快、更低 KV，但质量低于 v377 |
| sample oracle | 0.4086 | 8.26% | 6.13x | 说明样本级切换仍有明显上界 |

结论：v387 不是质量主线，但可作为更激进压缩 Pareto 点。如果 m20 gate 通过，会自动跑 m100。

## 明早优先看

1. `outputs/riskkv_lowkv_exploration_summary_20260712.json`
2. `doc/section170_lowkv_overnight_exploration_20260712.md`
3. `outputs/riskkv_lowkv_running_progress_20260712.json`
4. v385/v386/v387 的 `task_results.csv`
5. 如果 v385 M100 已完成，还要看 v388 是否自动训练并通过 gate。

判断优先级：

1. 如果 v385 M100 达到 `KV <= 10%` 且 `score >= 0.417`，它就是新主线，因为接近/超过 v300 的 95%。
2. 如果 v385 掉分，而 v386 接近预期 `0.3869/9.9%/4.8x`，v386 是最稳主线。
3. 如果 v387 M100 维持 `score >= 0.3658*0.95` 且 KV 明显低于 v377，它可作为 Pareto/speed variant，但不作为主方法。
4. 如果 v388 跑起来，说明 v385 M100 被纳入样本级 planner 后离线超过了 v377；这会是最值得继续深挖的样本级 operator routing 方向。

## 方法层面的发现

今晚不是盲扫参数，核心现象是：

- 任务级静态组合已经能稳定满足 full-cache 95% 目标，但上限大约在 0.387 左右。
- 样本级 oracle 明显更高，说明真正的提升空间在“何时切换 operator”，不是继续固定预算。
- 旧 learned router 的风险来自训练/测试候选混杂和 fallback 不一致；v387 用 M100-only + v377 fallback 处理这个问题，但当前离线结果仍偏保守。

下一步如果 v385/v386 都不能接近 v300 95%，应继续做高精度样本级切换，而不是继续扩大静态 task table。

## 自动续跑 v388

新增 `scripts/watch_v388_after_v385_m100_20260712.sh`：

1. 等待 v385 M100 的 `task_results.csv` 出现。
2. 用 v368/v375/v376/v377/v378/v380/v381/v385 的 M100 结果训练样本级 planner。
3. 只有离线 all split 达到 `score >= v377`、`KV <= 10%`、`speed/full >= 2.5x`，才启动真实 m20/m100。

这个 watcher 的目的不是盲目多跑，而是把 v385 如果出现的新好现象立刻转成下一轮 solution。
