# Section 152: v311 safe speed patch

日期：2026-07-11

## 动机

当前已完成的 partial M100 结果显示：

- b16 group2/group4 在 `narrativeqa`、`qasper`、`multifieldqa_en`、`2wikimqa` 上仍低于 v300，不能直接替换 QA 主线。
- bounded fallback 3k/4k 对 `narrativeqa`、`musique`、`2wikimqa` 掉分明显，也不能作为通用 QA fallback。
- `qmsum` 的 64-token decode 是较安全的 speed patch：v300 为 0.154638 / 14.71% KV / 2.1931s online，v305 qmsum64 为 0.150226 / 14.71% KV / 1.8338s online，分数约为 v300 的 97.1%，但 online 下降约 16.4%。
- `repobench-p` 的主要 KV 尾部来自 verifier retry 失败后的 full fallback，因此 v306 改为 bounded retry no-full，等待结果。

## v311 设计

v311 不是新的 QA 主线，而是保守 speed patch：

```text
v300 main policy
+ qmsum short_decode_max_tokens = 64
+ repobench-p bounded retry no full fallback
```

目标是先在不破坏 QA score 的情况下，降低全局 online 和 KV 尾部。

## 输出位置

组合 watcher：

```bash
scripts/watch_combine_v311_safe_speedpatch_20260711.sh
```

最终汇总：

```text
outputs/riskkv_v19_v311_safe_speedpatch_20260711/summary_table.csv
```

## 判据

- 如果 v306 repobench score 保持在 v300 的 95% 以上，并且 KV/online 明显下降，v311 可以作为当前最稳健 practical method。
- 如果 v306 repobench 掉分，v311 只保留 qmsum64 patch，repobench 继续沿用 v300。
