# Section 194: Auto Follow-up Queue for Composer Validation (2026-07-12)

## 新增自动接力

为了避免 M100 composer 完成后还需要手动启动 M200 稳定性验证，新增并启动了：

```bash
scripts/watch_launch_best_composer_m200_20260712.sh
```

它每 5 分钟检查 v430/v431/v435 的 M100 完成结果。如果某个候选满足：

```text
score >= 95% full
KV <= 10%
speed >= 2.5x
```

就自动选择 score 最高的候选，启动 LongBench M200 验证。

日志：

```text
outputs/logs/watch_launch_best_composer_m200_20260712.log
outputs/logs/watch_launch_best_composer_m200_20260712.select.log
```

## 当前候选队列

| 实验 | 状态 | 目的 |
|---|---|---|
| v430 M100 | running | constrained composer, KV<=6%, speed>=6x |
| v431 M100 | running | constrained composer, KV<=8%, speed>=5x |
| v435 M100 | queued | DP quality-oriented composer, KV<=10%, speed>=3.5x |
| best composer M200 | watcher running | M100 通过后自动启动 |

v435 已经排队等空卡。它的预测点是：

```text
score ~= 0.3920
KV ~= 8.71%
speed ~= 4.39x
```

这个点比 v430/v431 更偏质量优先，但仍满足用户目标的 1%-10% KV 和 2.5x+ speed。

## 当前 partial 观察

截至最近一次 dashboard：

```text
v430 partial: score ~= 0.3206, KV ~= 9.98%
v431 partial: score ~= 0.3215, KV ~= 10.05%
```

这个 partial 不能直接解释为最终结果，因为当前样本段集中在 hard QA，高 KV 任务占比偏高。后续 direct / code / short generation 任务会拉低平均 KV，并且提高平均 score。

## 明早读取顺序

优先看自动状态文件：

```bash
cat outputs/lowkv_queue_status_20260712.txt
```

如果 composer M100 已完成，再看自动接力日志：

```bash
tail -n 50 outputs/logs/watch_launch_best_composer_m200_20260712.log
tail -n 50 outputs/logs/watch_launch_best_composer_m200_20260712.select.log
```

如果 M200 已启动，输出目录形如：

```text
outputs/riskkv_v19_<best_label>_20260712_<best>_m200_auto_m200_bDyn_pDyn
```
