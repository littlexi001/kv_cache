# Section 153: overnight status and partial results

日期：2026-07-11

## 当前服务器状态

已提交并在服务器后台排队/运行：

| Sweep | 内容 | 汇总输出 |
|---|---|---|
| v301/v302 | b16 group4/group2 high-recall | `outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711/summary_table.csv` |
| v304/v305 | bounded fallback 4k/3k | `outputs/riskkv_v19_v304_v305_bounded_fallback_20260711/summary_table.csv` |
| v306 | repobench bounded retry no full fallback | `outputs/riskkv_v19_v306_repobench_retry_20260711/summary_table.csv` |
| v307/v308 | b16 pure-fine / pure-fine + window | `outputs/riskkv_v19_v307_v308_b16_purefine_sweep_20260711/summary_table.csv` |
| v309/v310 | micro-block locator + span repack | `outputs/riskkv_v19_v309_v310_b16_microspan_sweep_20260711/summary_table.csv` |
| v311 | v300 + qmsum64 + repobench bounded retry | `outputs/riskkv_v19_v311_safe_speedpatch_20260711/summary_table.csv` |

截至当前，GPU 已满载，runner 会自动等待空闲 GPU；watcher 已经挂起等待结果文件。

## 已完成 partial M100 结果

v300 对照：

| Task | v300 Score | v300 KV | v300 Online |
|---|---:|---:|---:|
| narrativeqa | 0.195960 | 36.16% | 0.2419s |
| qasper | 0.423567 | 43.64% | 0.4415s |
| multifieldqa_en | 0.569479 | 74.04% | 0.7445s |
| hotpotqa | 0.544469 | 54.91% | 0.2008s |
| 2wikimqa | 0.444420 | 38.95% | 0.2697s |
| musique | 0.255063 | 76.91% | 0.3782s |
| qmsum | 0.154638 | 14.71% | 2.1931s |
| repobench-p | 0.551255 | 46.40% | 2.3824s |

已完成的新单任务点：

| Run | Task | Score | KV | Online | 判断 |
|---|---|---:|---:|---:|---|
| v301 b16 group4 | 2wikimqa | 0.363778 | 42.90% | 0.2355s | 低于 v300，不可直接替换 |
| v301 b16 group4 | multifieldqa_en | 0.424819 | 35.30% | 1.2203s | KV 降了但掉分太多 |
| v301 b16 group4 | narrativeqa | 0.178302 | 21.04% | 0.2031s | 接近但低于 v300，需看 full sweep |
| v302 b16 group2 | 2wikimqa | 0.356333 | 43.62% | 0.2268s | 低于 v300 |
| v302 b16 group2 | multifieldqa_en | 0.422372 | 35.30% | 1.2187s | 低于 v300 |
| v302 b16 group2 | qasper | 0.347297 | 51.27% | 0.4222s | 低于 v300 |
| v304 bounded4k | 2wikimqa | 0.370677 | 30.50% | 0.3041s | KV 降但分数不够 |
| v304 bounded4k | musique | 0.148923 | 44.37% | 0.2787s | 掉分明显 |
| v304 bounded4k | narrativeqa | 0.144804 | 22.19% | 0.2577s | 掉分明显 |
| v305 bounded3k | 2wikimqa | 0.370677 | 27.55% | 0.3374s | KV 好但分数不够 |
| v305 bounded3k | musique | 0.158849 | 34.59% | 0.2408s | 掉分明显 |
| v305 bounded3k | narrativeqa | 0.143115 | 17.99% | 0.2088s | 掉分明显 |
| v305 qmsum64 | qmsum | 0.150226 | 14.71% | 1.8338s | 保留 97.1% 分数，online 降 16.4%，可作为 speed patch |
| v305 bounded3k | repobench-p | 0.528923 | 39.47% | 2.7265s | 分数约 95.9%，但 online 变慢；等待 v306 no-full retry |

## 当前判断

1. 不能把 b16 group2/group4 或 bounded fallback 直接作为 QA 主线；在 `qasper`、`multifieldqa_en`、`2wikimqa`、`musique` 上掉分太明显。
2. `qmsum64` 是目前已经验证的安全 speed patch，适合并入 v311。
3. repobench 的关键问题仍是 verifier retry 失败后 full fallback；v306 正在验证“bounded retry no full fallback”是否能压 KV 尾部。
4. 用户关于 b16 的假设仍值得继续验证，但要升级成 `micro-block locator + span repack`，而不是单纯多选 16-token block。v309/v310 正在跑这一点。

## 明早优先看什么

1. `outputs/riskkv_v19_v309_v310_b16_microspan_sweep_20260711/summary_table.csv`
   - 如果 v309/v310 明显好于 v307/v308，说明 micro-block locator + span repack 是新的论文主线候选。
2. `outputs/riskkv_v19_v311_safe_speedpatch_20260711/summary_table.csv`
   - 如果 v306 repobench 不掉分，v311 可能成为当前最稳 practical method。
3. `outputs/riskkv_v19_v307_v308_b16_purefine_sweep_20260711/summary_table.csv`
   - 用于判断 pure fine block 是否有独立价值，还是必须经过 span repack。
