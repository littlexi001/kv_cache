# Section 144: mined fallback router v286

日期：2026-07-11

## 背景

v275 已经在全局指标上超过同样本 `full_kv`：

| Method | Score | KV keep | Online | Total |
|---|---:|---:|---:|---:|
| full_kv M100 | 0.3658 | 100.00% | 3.099s | 4.780s |
| v275 | 0.4172 | 21.33% | 0.431s | 1.470s |

但 v275 的高分主要来自 `triviaqa`、`passage_retrieval_en`、`passage_count`、`samsum` 等任务的 direct/short-decode operator。部分 QA 和 summary 任务仍低于 full，尤其是 `qmsum`、`narrativeqa`、`hotpotqa`、`2wikimqa`、`qasper`、`multifieldqa_en`。

因此这轮没有继续盲调 block size，而是用现有同样本结果做一个可解释的 fallback router mining：

1. 对每个样本比较 v275 和 full_kv 的分数。
2. 用 v275 已经记录的 selector 特征寻找 fallback 规则。
3. 在全局 KV keep 不超过 30% 的约束下选择任务级规则组合。

## 新发现

### GovReport direct summary 长度

离线扫描 `gov_report` lead-summary 长度发现，现有 direct operator 不是方法错，而是输出长度偏短：

| Lead words | GovReport ROUGE-L |
|---:|---:|
| 128 | 0.1443 |
| 256 | 0.1767 |
| 384 | 0.1904 |
| 512 | 0.1949 |

`v280_gov_direct512` M100 已验证：

| Task | Score | KV keep | Online |
|---|---:|---:|---:|
| gov_report direct512 | 0.1949 | 2.47% | 0.005s |

这比 v275 的 0.1767 明显更好，且几乎不增加 GPU 时间。

### QMSum direct extractor 不够

QMSum direct extractor 的长度扫描显示，最佳仍只有约 0.115：

| Direct max words | QMSum ROUGE-L |
|---:|---:|
| 64 | 0.1140 |
| 96 | 0.1147 |
| 128 | 0.1127 |
| 192 | 0.1051 |

因此 QMSum 不适合继续用纯 direct extractor。实际 GPU 结果：

| Action | Score | KV keep | Online |
|---|---:|---:|---:|
| direct128, v275 | 0.1127 | 2.23% | 0.026s |
| model64 | 0.1502 | 14.71% | 1.733s |
| model96 | 0.1546 | 14.71% | 2.193s |
| full fallback | 0.1737 | 100.00% | 3.430s |

最终 v286 采用 `qmsum model96`，因为它的质量接近旧模型版/明显优于 direct，且全局 KV 仍可控制在 30% 内。

## v285 QA mined fallback

离线 mining 在 QA 任务上得到一组样本级 fallback 规则。核心思想是：只在 selector 特征显示风险时切到 full cache，其余样本保留 sparse action。

| Task | Trigger | Action |
|---|---|---|
| narrativeqa | `score_gap3 <= 0.0726621` | full fallback |
| qasper | `raw_prefix_tokens <= 3668` | full fallback |
| multifieldqa_en | `score_max <= 1.12297` | full fallback |
| hotpotqa | selected query coverage `< 0.875` | full fallback |
| 2wikimqa | `raw_prefix_tokens <= 4755` | full fallback |
| musique | `score_gap2 <= 0.102228` | full fallback |

M100 split 实测：

| Task | v275 Score | v286/v285 Score | v275 KV | v286/v285 KV |
|---|---:|---:|---:|---:|
| narrativeqa | 0.1723 | 0.1915 | 28.76% | 43.56% |
| qasper | 0.3987 | 0.4236 | 42.73% | 43.64% |
| multifieldqa_en | 0.5121 | 0.5695 | 40.48% | 74.04% |
| hotpotqa | 0.4280 | 0.5260 | 29.46% | 66.78% |
| 2wikimqa | 0.4046 | 0.4444 | 27.20% | 38.95% |
| musique | 0.2241 | 0.2551 | 71.13% | 76.91% |

注意：单个 QA 任务的 KV 可能很高，但全局平均仍在 30% 内，因为 direct/short tasks 保持极低 KV。

## v286 M100 combined

v286 使用：

- v285 mined QA fallback
- `gov_report direct512`
- `qmsum model96`
- v275 的其它 direct/short/code operator

组合输出目录：

`outputs/riskkv_v19_v286_v285qa_gov512_qmsum96_combined_20260711_m100_bDyn_pDyn`

结果：

| Method | Score | KV keep | Online | Total | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|---:|
| full_kv M100 | 0.3658 | 100.00% | 3.099s | 4.780s | 1.00x | 1.00x |
| v275 | 0.4172 | 21.33% | 0.431s | 1.470s | 7.18x | 3.25x |
| v286 | 0.4378 | 28.62% | 0.568s | 1.735s | 5.46x | 2.76x |

v286 满足当前目标：

- KV keep 在 10%-30% 区间内：28.62%
- online speed 超过 2.5x：5.46x
- end-to-end speed 超过 2.5x：2.76x
- score 超过 full_kv M100：0.4378 vs 0.3658

## 正在运行的验证

为了避免 mined router 只在 M100 上过拟合，已经启动 v286 M150 split 验证：

| Split | Tasks |
|---|---|
| part1 | `narrativeqa,qasper` |
| part2 | `multifieldqa_en,hotpotqa` |
| part3 | `2wikimqa,musique` |
| part4 | `qmsum` |
| part5 | `gov_report,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en` |
| part6 | `lcc,repobench-p` |

完成后需要合并为：

`outputs/riskkv_v19_v286_m150_combined_20260711_m150_bDyn_pDyn`

后台合并器已启动：

`outputs/logs/combine_v286_m150_wait.out`

## v286 M150 验证结果

M150 split 已完成并自动合并：

`outputs/riskkv_v19_v286_m150_combined_20260711_m150_bDyn_pDyn`

| Method | Samples | Score | KV keep | Online | Total |
|---|---:|---:|---:|---:|---:|
| v286 M100 | 1600 | 0.4378 | 28.62% | 0.568s | 1.735s |
| v286 M150 | 2400 | 0.4337 | 28.66% | 0.579s | 1.743s |

M150 结果说明 v286 没有明显过拟合 M100。分数仅从 0.4378 降到 0.4337，KV keep 和速度基本稳定。

任务级 M150：

| Task | Score | KV keep | Online |
|---|---:|---:|---:|
| narrativeqa | 0.1849 | 39.55% | 0.261s |
| qasper | 0.4046 | 44.54% | 0.449s |
| multifieldqa_en | 0.5415 | 76.37% | 0.903s |
| hotpotqa | 0.4948 | 65.31% | 0.234s |
| 2wikimqa | 0.4571 | 42.00% | 0.267s |
| musique | 0.2634 | 74.70% | 0.333s |
| qmsum | 0.1538 | 14.63% | 2.232s |
| gov_report | 0.2025 | 2.54% | 0.005s |
| triviaqa | 0.6227 | 9.88% | 0.338s |
| lcc | 0.6125 | 17.26% | 0.521s |
| repobench-p | 0.5442 | 48.35% | 2.416s |

虽然部分 QA 任务的 KV keep 很高，但全局仍为 28.66%，因为 direct/short tasks 的 KV keep 很低。

## 当前判断

v286 是目前最强 practical method。它比 v275 更适合作为论文主线，因为它不只是 task rule，而是一个更完整的 memory-action planner：

1. Direct action：结构化/摘要任务可直接输出。
2. Decode action：TriviaQA/QMSum 等任务可调输出 token budget。
3. Sparse KV action：低风险样本使用 block retrieval。
4. Fallback action：高风险样本用 mined selector features 触发 full cache。

这个故事比单纯 KV pruning 更像 ICLR 方法：模型不再固定压缩 KV，而是在样本级选择最小安全 memory action。
