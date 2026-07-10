# Section 124: LongBench 更激进压缩实验记录（2026-07-10）

## 目标

在 RULER 效果已经满意的前提下，继续压 LongBench 的 KV token 占比，同时尽量保持 LongBench 分数不低于 full KV baseline 的 95%。

本轮使用 Llama-3.1-8B-Instruct，LongBench 16 tasks，每个 task 20 samples，和前面 v138/full baseline 使用同一组样本。

full KV baseline：

| 方法 | Score | KV keep | Online | Speedup |
|---|---:|---:|---:|---:|
| full_raw | 0.372655 | 100.00% | 3.033s | 1.00x |

95% full 阈值为 0.354022。

## 关键结果

| 方法 | Score | KV keep | Online | Speedup vs full |
|---|---:|---:|---:|---:|
| v138 speed frontier | 0.354193 | 29.61% | 1.203s | 2.52x |
| v144 selective safe | 0.356972 | 26.24% | 1.208s | 2.51x |
| v145 qasper1024 | 0.355646 | 25.54% | 1.217s | 2.49x |
| v146 ultra safe | 0.361040 | 23.87% | 1.190s | 2.55x |
| v148 multihop1536 | 0.361040 | 23.45% | 1.193s | 2.54x |
| v150 multihop768 | 0.359478 | 22.83% | 1.200s | 2.53x |
| v151 hotpot768/musique1536 | 0.362603 | 23.15% | 1.208s | 2.51x |
| v152 hotpot512/musique1536 | 0.361040 | 23.05% | 1.186s | 2.56x |
| v153 hotpot768/musique1280 | 0.362603 | 23.04% | 1.210s | 2.51x |
| v154 hotpot768/musique1152 | 0.359478 | 22.99% | 1.211s | 2.51x |
| v155 musique1280/qasper640 | 0.359115 | 22.87% | 1.432s | 2.12x |

当前主推版本更新为 v153：Score 0.362603，约为 full baseline 的 97.30%，KV keep 23.04%，online speedup 2.51x。

如果优先压缩率，v150 可以作为低 KV 备选：Score 0.359478，KV keep 22.83%，仍高于 95% full 阈值。

## 失败路线

| 方法 | Score | KV keep | 结论 |
|---|---:|---:|---|
| v139 bounded multihop1536 | 0.334107 | 20.22% | 低于 95% full，不能作为主方法 |
| v140 bounded multihop1024 | 0.321136 | 16.21% | 压缩太狠 |
| v141 extreme longbench20kv | 0.307516 | 12.13% | 质量断崖 |
| v142 bounded allrisk1536 | 0.321921 | 17.75% | 限制 full fallback 会伤 QA |
| v143 aggressive KV no decode cap | 0.314652 | 14.97% | 去掉 decode cap 也救不回过度压缩 |

主要结论：LongBench 不能简单追求 20% 以下 KV keep。HotpotQA、MuSiQue、PassageCount 是压缩断崖任务；如果把它们的 full fallback 限死到 1536/1024，会明显掉分。

## v151 策略

v151/v153 的核心是选择性压缩，而不是全局降预算：

| Task family | 动作 |
|---|---|
| HotpotQA | base budget 768，但保留 score-risk full fallback |
| MuSiQue | v151 为 base budget 1536；v153 降到 1280，保留 score-risk full fallback |
| PassageCount | 保持 2048 + structured fingerprint，不再压 |
| NarrativeQA / MultiFieldQA | 保持 entropy risk full fallback |
| Qasper | budget 768 + task bridge flow |
| GovReport / QMSum / MultiNews / SAMSum | 降到 384/384/384/384，保留 summarization decode cap |
| TREC | budget 384，risk fallback 到 1024 |
| PassageRetrieval | budget 768 + IDF/label support/short decode |
| RepoBench-P | recent 448，budget 512 |

## 任务级观察

v151 中比较重要的任务级结果：

| Task | Score | KV keep | 备注 |
|---|---:|---:|---|
| HotpotQA | 0.385833 | 70.12% | 比 v138/v148 更高，说明低 base budget 能去掉部分干扰，但 full fallback 必须保留 |
| MuSiQue | 0.300000 | 60.41% | 1536 base budget 还能保持质量，768/1024 会掉到 0.25 |
| Qasper | 0.506869 | 18.79% | 768 budget 仍可用；512 会掉到 0.427 |
| PassageCount | 0.125000 | 28.58% | 不能继续压，1024/1536 会明显掉分 |
| RepoBench-P | 0.483519 | 10.57% | 512/recent448 反而比 v138 更好 |
| TREC | 0.750000 | 15.80% | 小预算 + 1024 fallback 优于 v138 |

## 当前判断

更新：m50 验证推翻了“v151/v153 可直接作为最终主方法”的判断。m20 上 v151/v153 很好，但 m50 上分数明显下降，说明 m20 存在样本偶然性/过拟合。

m50 baseline：

| 方法 | Score | KV keep | Online | 备注 |
|---|---:|---:|---:|---|
| full_raw m50 | 0.371970 | 100.00% | 3.283s | 95% 阈值为 0.353371 |
| v150 m50 | 0.314652 | 23.49% | 1.222s | 低 KV 候选失败 |
| v151 m50 | 0.318485 | 23.73% | 1.315s | 低 KV 主候选失败 |
| v153 m50 | 0.317235 | 23.65% | 1.227s | MuSiQue 1280 不稳定 |

v153 在 m20 上比 v138 同时提升质量和压缩率：

- Score: 0.354193 -> 0.362603
- KV keep: 29.61% -> 23.04%
- Online speedup: 2.52x -> 2.51x，基本不变

但这个结论只能作为 m20 探索结果，不能作为 paper 主表结论。

方法故事上应强调“risk-aware selective compression”：不是所有任务都同等可压，而是用风险门控把任务分成 evidence-fragile 和 compression-tolerant 两类。HotpotQA/MuSiQue/PassageCount 需要保留 fallback；Qasper/TREC/RepoBench/summarization 可以更激进。

补充探针结论：

- MuSiQue 1280 可以保持 0.300000 分，并把全局 KV 从 v151 的 23.15% 轻微降到 23.04%。
- MuSiQue 1152 会把总分降到 0.359478，说明 1280 附近是当前安全边界。
- Qasper 640 会把 Qasper 从 0.506869 降到 0.451069，不能作为主方法；Qasper 768 是当前安全下限。

m50 任务级失败点：

| Task | full m50 | v151 m50 | 主要问题 |
|---|---:|---:|---|
| Qasper | 0.4773 | 0.3780 | 768 budget 在 m50 不稳 |
| 2WikiMQA | 0.4510 | 0.3090 | 512 budget 不稳 |
| PassageRetrieval | 0.7200 | 0.6100 | 768 + short decode 不够 |
| RepoBench-P | 0.5330 | 0.4269 | 512 recent-only 不够 |
| TREC | 0.7000 | 0.6000 | 小预算 risk fallback 不够 |
| PassageCount | 0.1240 | 0.0500 | structured fingerprint 在 m50 失败 |
| GovReport | 0.2144 | 0.0968 | decode cap 明显伤 ROUGE |

当前新的判断：要同时做到 m50 过 95% 和 KV < 30%，不能再靠 task-level 静态表。需要 per-example risk router，只对 Qasper/2Wiki/PassageRetrieval/RepoBench/TREC/PassageCount 中的危险样本 fallback。

正在跑的后续实验：

| 实验 | 目的 |
|---|---|
| v158 m50 | 去掉 summarization short decode cap，分清 KV 质量和输出长度截断 |
| v159 m50 | 质量恢复上界，估算过 m50 需要多少 fallback 成本 |
| v150/v151/v153 m100 | 验证 m20/m50 结论是否随样本数继续稳定 |

## 下一步

1. 等 v158/v159/m100 完成，确定 m50 的质量-速度-KV Pareto 边界。
2. 用 m50 task-level 失败点蒸馏 per-example router，而不是继续扩大 task table。
3. PassageCount 需要单独做结构化计数 verifier/metadata KV，否则它会限制低 KV 稳定性。
4. 论文主表不能使用 output decode cap 混淆 KV 压缩质量；应把质量评测和 online speed/生成长度策略分开报告。
