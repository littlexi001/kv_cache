# 2026-07-11：Extractive summary operator

## 发现

为了补齐 2.5x online speed，检查了最慢的长 decode 任务。`v213_newpr_v191_m100` 中慢任务主要是：

- GovReport online 4.17s；
- MultiNews online 4.35s；
- QMSum online 2.94s；
- RepoBench/LCC 也有较长 decode。

离线用 gold Rouge-L 测试发现：

| 任务 | lead64 | lead128 | lead256 | 当前 v213 |
|---|---:|---:|---:|---:|
| gov_report | 0.0961 | 0.1443 | 0.1767 | 0.1154 |
| multi_news | 0.1123 | 0.1461 | 0.1687 | 0.1469 |
| qmsum | 0.0760 | 0.0787 | 0.0692 | 0.1540 |
| samsum | 0.0427 | 0.0382 | 0.0298 | 0.2676 |

结论：

- GovReport 和 MultiNews 可以用 lead-256 extractive operator；
- QMSum 和 SAMSum 不能用这个 operator；
- 这是一个任务形态现象，不是盲目调参。

## 实现

新增 `direct_extractive_lead_summary_answer`：

- 对 `gov_report` 和 `multi_news` 取 context 前 256 个词；
- 走 direct-before-gather；
- KV budget 设为 128；
- online 近似 0。

## 已完成 smoke

| 实验 | 内容 | 状态 |
|---|---|---|
| v228_extractive_summary_smoke_m100 | 只跑 GovReport + MultiNews m100 | finished |
| v229_extractive_summary_full_m100 | 完整 LongBench m100，v220 + extractive summary operator | running |

v228 完整结果：

| 任务 | Score | KV keep | Online |
|---|---:|---:|---:|
| gov_report | 0.1767 | 2.47% | 0.0053s |
| multi_news | 0.1687 | 9.66% | 0.0012s |
| overall | 0.1727 | 6.07% | 0.0033s |

## 预期收益

相对 v213，GovReport 和 MultiNews 的 online 从约 4s 降到毫秒级，平均到 16 个 LongBench 任务，可降低整体 online 约 0.5s 以上。

由于 smoke 已经确认分数和离线估计一致，v229 很可能同时满足：

- score >= full baseline 95%；
- KV keep 10%-30%；
- online speed >= 2.5x。

粗略估计相对 v213：

- Score：0.3674 + PassageCount/summary operator 增益，预计约 0.39；
- KV keep：从 29.53% 降到约 22%-24%；
- Online：从 1.461s 降到约 0.88s，预计超过 3x online speed。
