# Section 160: qasper BM25 bridge budget follow-up

日期: 2026-07-11

## 背景

BM25-bridge smoke 的整体结论偏负, 但 qasper 出现明显正信号:

| Method | Samples | Score | vs full | vs v300 | KV |
|---|---:|---:|---:|---:|---:|
| v314 B16 BM25-bridge | 20 | 0.5346 | 118.5% | 126.2% | 46.94% |
| v315 B128 BM25-bridge | 20 | 0.5112 | 113.3% | 120.7% | 35.68% |

问题是 KV 仍然高于 30%。因此本轮只针对 qasper 做预算压缩, 不再全任务扫。

## 新实验

| Version | 设计 |
|---|---|
| v318 | qasper, B=128, BM25-bridge, budget=1280 |
| v319 | qasper, B=128, BM25-bridge, budget=1024 |

样本:

```text
M20 qasper
```

## 判据

如果 v318/v319 能保持 qasper M20 score 明显高于 v300 且 KV <= 30%, 则把该 qasper-specific action 纳入下一版 practical router。

如果 1024 明显掉分但 1280 仍高分, 可以做 qasper 的 dynamic budget ladder: 低风险 1024, 高风险 1280/1536。

如果两者都掉分, qasper 的 BM25-bridge 只能作为高 KV ablation, 不纳入主线。
