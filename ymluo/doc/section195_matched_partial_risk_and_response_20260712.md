# Section 195: Matched Partial Risk and Response (2026-07-12)

## 为什么新增 matched partial

之前 live partial 直接按日志已跑样本求均值，但不同实验进度不同，不能直接比较。进一步检查发现 runner 当前使用 `--log_every 20`，日志只每 20 条打印一次 sample id，因此 sample-id matched compare 对现有日志不可靠。

为此新增：

```bash
scripts/summarize_live_matched_metrics_20260712.py
```

当前对 M200 使用 `--match ordinal`。它假设 v427/v428/full M200 的样本顺序一致，用全局样本序号做匹配，比直接比较 partial 均值更可靠。

## 当前 matched partial 现象

截至当前读取：

| Run | Matched n | Score | Full score | Retention | KV | Speed |
|---|---:|---:|---:|---:|---:|---:|
| v427 M200 partial | 883 | 0.2864 | 0.4353 | 65.79% | 9.79% | 1.28x |
| v428 M200 partial | 869 | 0.2845 | 0.4337 | 65.61% | 9.73% | 1.35x |

这个不是最终 M200 结论，因为当前 matched prefix 集中在 narrativeqa/qasper/multifield/hotpot/2wikimqa 等 hard QA 任务，后续 retrieval/direct/code/short-generation 会改变平均值。

但它暴露了一个真实风险：

1. v427/v428 虽然 M100 已经很好，但在 M200 的 hard-QA prefix 上 retention 明显不足；
2. online speed 在这些任务上也只有约 1.3x，不满足最终目标；
3. 因此不能只拿 v427/v428 M100 作为 ICLR 级完整证据。

##  대응策略

当前已经排队的实验正是针对这个风险：

| 实验 | 作用 |
|---|---|
| v430/v431 M100 | constrained composer，检查自动 source selection 是否改善 hard QA 同时控制 KV |
| v435 M100 | quality-oriented DP composer，允许平均 KV 到 10%，优先补 hard-QA 质量 |
| best-composer M200 watcher | 如果 v430/v431/v435 任一通过 M100 gate，自动启动 M200 稳定性验证 |

v435 的预测为：

```text
score ~= 0.3920
KV ~= 8.71%
speed ~= 4.39x
```

它比 v430/v431 更适合应对 hard-QA retention risk，因此已加入队列。

## RULER matched partial

RULER v427 partial ordinal matched：

```text
score retention ~= 90.99%
KV ~= 8.14%
speed ~= 0.31x
```

这说明 RULER 当前主要问题不是 KV 平均值，而是 online overhead 和 4k 子任务局部 KV > 10%。v436 low-KV RULER 补测已经排队，用来确认更小 wildcard budget 是否能把 4k KV 压回目标区间。

## 当前判断

目前不能把目标标记完成。已完成 M100 证明方法有潜力，但 ICLR 需要：

1. best composer 的完整 M100；
2. matched or full M200 stability；
3. RULER v436 的速度/KV修复结果；
4. ablation：source-preserving vs overlay, composer vs single frontier, DP composer vs Lagrange composer。
