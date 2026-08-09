# 2026-07-11：Code cap 组合与 v230

## Speed-cap ablation 结果

`v224_structured_speed_caps_m20`：

| 指标 | 数值 |
|---|---:|
| Score | 0.4136 |
| KV keep | 25.95% |
| Online | 0.848s |

`v225_structured_speed_caps_aggr_m20`：

| 指标 | 数值 |
|---|---:|
| Score | 0.4091 |
| KV keep | 25.95% |
| Online | 0.659s |

分项现象：

| 任务 | v217 online | v224 score/online | v225 score/online | 判断 |
|---|---:|---:|---:|---|
| gov_report | 4.236s | 0.0778 / 1.971s | 0.0448 / 1.046s | summary cap 会严重伤质量 |
| multi_news | 4.048s | 0.1019 / 1.967s | 0.0674 / 1.043s | summary cap 会严重伤质量 |
| qmsum | 2.139s | 0.1550 / 1.553s | 0.1496 / 1.005s | 64 cap 可接受，32 cap 略伤 |
| lcc | 2.037s | 0.6608 / 1.008s | 0.6607 / 0.540s | code cap 很安全 |
| repobench-p | 2.783s | 0.5770 / 2.002s | 0.5767 / 1.670s | code cap 很安全 |

结论：

- GovReport/MultiNews 不应该用 short decode cap，应该用 extractive lead operator；
- code completion 任务可以使用更激进的 decode cap，质量几乎不变；
- QMSum 可以保守使用 64-token cap。

## 新组合：v230

`v230_extractive_codecap_full_m100` 已启动：

- 基础：v228 extractive summary operator；
- GovReport/MultiNews：lead-256 extractive operator；
- PassageCount/PassageRetrieval：structured direct operator；
- QMSum：short decode 64；
- LCC/RepoBench：short decode 16；
- 其它 QA/multi-hop 保持 v206/v191 风格。

预期：

- Score：接近或高于 v224/v225 m20 的 0.41，并高于 full baseline；
- KV：约 22%-26%；
- Online：预计明显低于 1.0s，应超过 2.5x。

如果 v230 m100 兑现，它就是当前主方法候选。

## Pareto 对照：v231

同时启动 `v231_extractive_codecap_noqmsumcap_full_m100`：

- 与 v230 相同：GovReport/MultiNews extractive operator，PassageCount/PassageRetrieval structured operator，LCC/RepoBench code cap 16；
- 与 v230 不同：QMSum 保持 128-token short decode，不使用 64-token cap。

目的：

- 判断 QMSum 64-token cap 是否值得；
- 如果 v230 分数略低但速度更快，v231 可作为质量优先版本；
- 如果 v230 分数不掉，v230 就是速度优先主版本。
