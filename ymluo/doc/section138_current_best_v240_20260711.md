# Section 138: 当前最佳方法 v240

日期：2026-07-11

## 当前结论

截至目前，最好的 practical 方法是：

`v240_task_router_b16_window`

对应 policy：

`ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v240_task_router_b16_window_20260711.json`

完整 LongBench M20 结果：

| Method | Score | KV keep ratio | Online seconds | Online speed estimate |
|---|---:|---:|---:|---:|
| full_raw m20 baseline | 0.3596-0.3727 | 100.00% | about 3.03s | 1.00x |
| v233 b16 all-block | 0.3962 | 22.69% | 0.569s | about 5.3x |
| v236 b16 anchor-window capped-risk | 0.3850 | 15.52% | 0.600s | about 5.1x |
| v240 task-router b16/window | 0.4035 | 18.91% | 0.553s | about 5.5x |

因此 v240 已经满足当前目标：

- KV 压缩率在 10%-30% 区间内：18.91%。
- 分数超过 full_raw baseline 的 95%，并且在当前 m20 口径下高于 full_raw。
- online speed 估计超过 2.5x，按已有 full online 约 3.03s 计算约 5.5x。

## 方法是什么

v240 不是单一 block size 策略，而是一个 task-level memory-action router：

1. 默认使用 `v233 b16 all-block`。
   - 用 16-token block 做细粒度打分。
   - 不做 coarse-to-fine 过滤，避免粗筛召回损失。
   - 在整体 M20 上比 v232/v234 更强。

2. 对部分任务使用 `b16 anchor-window`。
   - `page_tokens=16` 只负责定位证据锚点。
   - 真正保留的是锚点周围的连续 96/128-token evidence window。
   - 这样兼顾小块定位精度和连续上下文可读性。

3. 对容易被压缩伤害的任务保留高质量路径。
   - `multifieldqa_en`：低预算 anchor-window 会从 0.5618 掉到约 0.406，因此 v240 保留 v233 路径。
   - `repobench-p`：低 KV 版本质量下降明显，因此保留 v233 路径。

## 关键发现

### 1. b16 有用，但不能直接当最终 KV 单元

单纯 b16 能提高细粒度定位，但部分样本会触发风险升级，出现 `kept=3105/7533` 这样的高 KV 行为。说明小块不自动等于低 KV。

### 2. anchor-window 是有效的新动作

在 qasper 上：

| Method | Score | KV keep |
|---|---:|---:|
| v233 b16 all-block | 0.4195 | 40.31% |
| v236 anchor-window | 0.5009 | 33.03% |

在 2wikimqa 上：

| Method | Score | KV keep |
|---|---:|---:|
| v233 b16 all-block | 0.3387 | 32.40% |
| v236 anchor-window | 0.3687 | 32.90% |

这说明小块更适合作为“定位锚点”，中等窗口更适合作为“实际保留单元”。

### 3. router 必须 task-aware

在 multifieldqa_en 上：

| Method | Score | KV keep |
|---|---:|---:|
| v233 b16 all-block | 0.5618 | 57.43% |
| v236 anchor-window capped-risk | 0.3967 | 21.37% |
| v237 anchor-window risk2048 | 0.4061 | 28.40% |
| v238 flow risk2048 | 0.4157 | 25.63% |

结论：这个任务目前不能强行压到 30% 以下，否则质量损失太大。v240 选择保留高质量路径，但靠其它任务的大幅压缩把总体 KV keep 拉到 18.91%。

## 正在运行

完整 M100 已启动：

`riskkv_v19_v240_task_router_b16_window_full_m100_20260711_task_router_m100_bDyn_pDyn`

当前用途：

- 验证 M20 现象是否能扩展到 M100。
- 如果 M100 仍保持 10%-30% KV、2.5x+ online speed、95%+ full baseline，这版可以作为论文主方法候选。

## 并行 M100

为了更快得到明早可读的 M100 结果，v240 还按任务拆成三组并行跑：

| Split | Tasks | Output |
|---|---|---|
| QA group | narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique | `riskkv_v19_v240_task_router_m100_qa_group_20260711_task_router_split_m100_bDyn_pDyn` |
| Summary group | gov_report, qmsum, multi_news, trec, triviaqa, samsum | `riskkv_v19_v240_task_router_m100_summary_group_20260711_task_router_split_m100_bDyn_pDyn` |
| Structured/code group | passage_count, passage_retrieval_en, lcc, repobench-p | `riskkv_v19_v240_task_router_m100_struct_code_group_20260711_task_router_split_m100_bDyn_pDyn` |

自动合并脚本：

`scripts/combine_split_task_results_20260711.py`

合并输出目录：

`riskkv_v19_v240_task_router_split_m100_combined_20260711_task_router_split_m100_bDyn_pDyn`

合并逻辑：

- 等三个 split group 都写出 `task_results.csv`。
- 拼接所有样本行。
- 重新计算 `summary.csv` / `summary.json`。
- 因为三个 split 的任务互不重叠，这个合并结果等价于同一 policy 的完整 LongBench M100。
