# Section 193: Live Partial Status and Decisions (2026-07-12)

## 已完成结论

LongBench M100 已完成的可靠结果仍然是：

| 方法 | Score | KV ratio | Online speed |
|---|---:|---:|---:|
| v427 Source-Preserving Frontier Routing | 0.3774 | 5.09% | 7.68x |
| v428 v427 + RepoBench source | 0.3813 | 6.68% | 6.81x |

这两个点已经满足用户目标：1%-10% KV、2.5x+ speed、95%+ full baseline。

## 当前运行中队列

统一 dashboard 脚本已经同步到服务器：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_lowkv_queue_20260712.py
```

live log parser 也已同步：

```bash
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_live_log_metrics_20260712.py --runs v427_m200,v428_m200,full_m200,v429_m100,v430_m100,v431_m100 --by-task
```

另外已经启动轻量 watcher，每 5 分钟刷新一次：

```text
outputs/lowkv_queue_status_20260712.txt
```

对应脚本：

```bash
scripts/watch_lowkv_queue_status_20260712.sh
```

运行中任务：

| 实验 | 当前用途 |
|---|---|
| v427/v428/full LongBench M200 | 检查 M100 结论是否稳定到更大样本 |
| v429 M100 | 质量优先 source-best-frontiers 验证 |
| v430/v431 M100 | constrained source composer 真实验证 |
| v433/v434 M100 | DP exact composer ablation，已排队等空卡 |
| full/v427 RULER M50 | synthetic retrieval 基线和风险暴露 |
| v436 RULER M50 | 低预算 RULER 修复，已排队等空卡 |

## Live partial 现象

截至当前读取日志时，v430/v431 已经跑过 narrativeqa、qasper、multifieldqa_en 的完整或接近完整段，并开始 hotpotqa。partial 指标不能当最终平均分，因为任务顺序强烈影响均值，但可以用来判断 source transfer 是否失效。

关键观察：

1. v430/v431 在 qasper 上 partial score 约 0.3297，高于原先基于 M100 预测的 0.3057；
2. multifieldqa_en partial score 约 0.4017，接近预测的 0.3995；
3. narrativeqa partial score 约 0.1536，高于 composer 里使用的 v417/v397 完成 M100 统计；
4. v430/v431 当前 KV ratio 偏高约 8.6%，是因为当前任务段集中在 QA 高预算任务；最终平均预计会被后续 direct/summary/code 任务拉回。

因此目前没有证据说明 composer 失效，反而 partial by-task 支持 v430/v431 正在兑现。

## RULER 风险确认

RULER v427 partial：

| 子任务段 | Score | KV ratio | Online |
|---|---:|---:|---:|
| niah_single_1_4096 | 0.8814 | 13.01% | 2.79s |
| niah_single_2_4096 | 0.9833 | 11.68% | 0.67s |
| 8192/16384 段 | 约 0.92-1.00 | 约 3%-6% | 0.35-2.50s |

这说明 RULER 上存在一个真实问题：因为 RULER 任务名不命中 LongBench task policy，v427 回落到 wildcard，4k 下 KV ratio 超过 10%，并且 online 可能慢于 full。

所以 v436 不是盲调参，而是针对 task-name fallback 暴露的问题做的修复：

```text
budget_tokens=224, sink_tokens=32, recent_tokens=32, page_tokens=64
```

目标是在 RULER 4k 下把 KV ratio 压回 10% 附近，同时观察 synthetic retrieval 质量是否还能维持。

## 当前判断

如果最终 v430/v431 M100 结果接近预测，则主方法建议定为：

```text
Source-Preserving Frontier Routing + Constrained Source Composer
```

如果 v433/v434 DP exact 与 v430/v431 接近，则论文里优先讲 DP composer，把 Lagrange composer 作为开发过程或弱化处理。

下一步必须等完整 CSV 后做：

1. M100 final table；
2. M200 stability table；
3. RULER length table；
4. source-preserving vs overlay ablation；
5. composer vs single-frontier ablation。
