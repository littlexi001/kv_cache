# Section 122: LongBench aggressive compression with structured fingerprint

## 目标

RULER 方向已经可以在长上下文定位任务上取得较好的压缩和加速信号；当前短板是 LongBench：如果直接压低预算，HotpotQA、MuSiQue、PassageCount、RepoBench-P 等任务会明显掉分。因此本轮目标不是追求统一低预算，而是在 LongBench 上寻找更激进但仍可用的实际策略。

核心判断：

- RULER 主线继续保留 v101/v93 类 length-aware localization policy。
- LongBench 不应套用 RULER 的固定低预算策略。
- LongBench 需要 task-conditioned minimum safe action：对可压任务 aggressive，对已验证高风险任务保护质量。

## 新增机制：structured fingerprint

本轮新增 `structured_fingerprint` action，主要用于 PassageCount。

动机是 PassageCount 的输入通常带有 `Paragraph k` / `Passage k` 这类结构标签，任务目标是统计去重后的段落数量。普通 query-block lexical scorer 在这个任务上不稳定，因为 query 本身没有足够的内容锚点；完全 fallback 又会让 KV keep 接近 100%。

`structured_fingerprint` 的做法：

1. 在每个 page/block 中识别结构标签，例如 `Paragraph 1`、`Passage 7`。
2. 对每个唯一结构标签，只保留第一次出现的 page 作为 fingerprint page。
3. fingerprint pages 最多使用 `budget * structured_fingerprint_budget_fraction` 的预算。
4. 剩余预算继续交给原有 page scorer / flow scorer。

这个动作不是 oracle，也不读取答案；它只利用输入文档自身的结构标签。

## 当前主要策略

### v111 aggressive fingerprint

配置：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v111_aggressive_fingerprint_20260710.json
```

主要变化：

| 任务 | 动作 |
| --- | --- |
| HotpotQA | 保持 full fallback，暂不激进压缩 |
| MuSiQue | 保持 full fallback，暂不激进压缩 |
| Qasper | 保持 v81 的 2048 budget + bridge |
| TREC | 释放到 512 budget，并保留 score-risk escalation |
| PassageCount | 新增 structured fingerprint，2048 budget，无 full fallback |
| RepoBench-P | 1024 budget，其中 recent tokens 960 |
| GovReport/QMSum/MultiNews | 1024 budget |
| LCC/SAMSum/TriviaQA 等 | 默认 512 budget |

### v112 aggressive fingerprint + Qasper1024

v112 进一步把 Qasper 从 2048 降到约 1024/1843 kept token 区间，目标是更省 KV；但实际收益很小，质量损失更明显。

### v113 PassageCount probe

v113 只测 PassageCount structured fingerprint，不使用 full fallback，用来验证这个新 action 是否真的有效。

## 结果

LongBench m20，实际可用策略，非 oracle：

| 方法 | LongBench score | KV keep | 平均 kept tokens | 备注 |
| --- | ---: | ---: | ---: | --- |
| full KV | 0.372655 | 100.00% | - | baseline |
| v81 quality branch | 0.380371 | 58.31% | - | 当前质量最稳 |
| v107 aggressive release | 0.359801 | 45.35% | - | 更省，但掉分较多 |
| v111 aggressive fingerprint | 0.368951 | 44.49% | 2747 | 当前 LongBench aggressive 首选 |
| v112 aggressive Qasper1024 | 0.366963 | 44.19% | 2734 | 比 v111 只省 0.30% KV，但掉分更多 |

结论：

- v111 相比 v107：score 从 0.3598 提升到 0.3690，同时 KV keep 从 45.35% 略降到 44.49%。
- v111 相比 v81：KV keep 从 58.31% 降到 44.49%，score 从 0.3804 降到 0.3690。
- v112 不值得作为当前主策略：它只比 v111 少约 0.3% KV keep，但 LongBench score 更低。

## v111 分任务结果

| task | score | KV keep | 说明 |
| --- | ---: | ---: | --- |
| 2WikiMQA | 0.4687 | 45.56% | 可压缩多跳，仍保留 verifier/risk action |
| HotpotQA | 0.4008 | 100.00% | 当前 scorer 低预算不安全 |
| MuSiQue | 0.3000 | 100.00% | 当前 scorer 低预算不安全 |
| Qasper | 0.5331 | 46.94% | 2048 + bridge 比 1024 更稳 |
| PassageCount | 0.1250 | 28.58% | structured fingerprint 生效 |
| PassageRetrieval | 0.6500 | 95.70% | label support/output verifier 仍触发较多 |
| RepoBench-P | 0.4090 | 20.98% | recent-heavy code action 有效压缩 |
| TREC | 0.7000 | 29.38% | 512 + score-risk 可用 |
| GovReport | 0.1789 | 16.12% | summarization 可压缩 |
| QMSum | 0.1562 | 14.57% | summarization 可压缩 |
| MultiNews | 0.1657 | 59.01% | 原始输入较短，比例不如绝对 token 直观 |

## PassageCount 单项验证

v113 PassageCount no-fallback probe：

| 方法 | PassageCount score | KV keep |
| --- | ---: | ---: |
| v107 类普通低预算策略 | 0.0100 | 28.58% |
| v113 structured fingerprint | 0.1250 | 28.58% |
| full/v81 参考 | 约 0.1500 | 100.00% |

这说明 PassageCount 不是必须 full fallback；它需要的是结构感知压缩，而不是普通 query lexical retrieval。

## 当前判断

现在建议把方法拆成三个实际分支，而不是一个统一 policy 强行覆盖所有任务：

| 分支 | 用途 | 当前代表 |
| --- | --- | --- |
| quality branch | LongBench 稳质量 | v81 |
| aggressive LongBench branch | LongBench 更低 KV、可接受掉分 | v111 |
| long-localization branch | RULER 长上下文定位 | v101/v93 |

论文故事可以从这里推进：

> RiskKV-Block is not a fixed-budget compressor. It is a risk-conditioned action planner that chooses the minimum safe KV action according to task structure, evidence risk, and context geometry.

v111 的新增价值在于引入了结构感知 action：对 counting / structured document task，不再用普通 query overlap，而是用 document-internal structural fingerprint 作为可压缩证据。

## 下一步

1. 将 v111 作为 LongBench aggressive branch 的当前首选。
2. 用 v81、v111、v101/v93 生成 action labels，训练一个 router 学会在 quality / aggressive / localization 三个分支之间选择。
3. 继续攻 HotpotQA 和 MuSiQue，但不要再简单堆 graph bridge；之前 v103-v109 已经证明收益差。下一轮更应该探索 entity-chain packet、support-set closure 或 answer-type constrained verifier。
4. 对 PassageRetrieval 的 95.7% keep 做专项压缩，它现在还是 LongBench 中最浪费的非 full-fallback 任务之一。
5. 把 structured fingerprint 写入论文 method，作为 task-structure-aware KV action 的例子。

## 追加：PassageRetrieval aggressive probe

v111 的 PassageRetrieval 分支仍然保留 95.70% KV，虽然 score 为 0.65，但明显不是理想压缩点。检查 `task_results.csv` 后发现：

- 20 个样本中 19 个触发 `output_fallback_active`。
- `grounding_verifier` 没有触发。
- 主要原因是 sparse decode 输出不符合 `Paragraph k` 格式，导致 output verifier 回退 full KV。

因此 PassageRetrieval 的问题不是简单预算不足，而是 sparse context 下模型容易输出解释性文本。

专项 probe 结果：

| 方法 | PassageRetrieval score | KV keep | online seconds | 结论 |
| --- | ---: | ---: | ---: | --- |
| v111 原分支 | 0.65 | 95.70% | 0.665 | 质量可用，但几乎全量 KV |
| v114 1024 no verifier | 0.15 | 13.96% | 0.989 | 太多 verbose answer，不能用 |
| v115 2048 no verifier | 0.20 | 27.57% | 0.975 | 加预算不能解决格式问题 |
| v116 1024 short decode | 0.05 | 13.96% | 0.323 | 只截短输出更差 |
| v117 2048 short decode | 0.10 | 27.57% | 0.333 | 只截短输出更差 |
| v118 1024 no-chat + short decode | 0.60 | 13.91% | 0.323 | 接近 v111 质量，KV 大幅降低 |
| v119 2048 no-chat + short decode | 0.60 | 27.52% | 0.327 | 不比 1024 更好 |

关键结论：

- `no-chat + short_decode` 是有效动作，说明 PassageRetrieval 的主要瓶颈是输出格式控制，而不是 evidence block retrieval。
- v118 是更好的 PassageRetrieval action：score 只比 v111 少 0.05，但 KV keep 从 95.70% 降到 13.91%。
- 2048 budget 没有比 1024 提升，说明这个任务不需要更多 block，而需要更强格式控制。

已启动 v120 全量 LongBench：

```text
configs/riskkv_task_policy_v120_aggressive_fingerprint_pr_shortdecode_20260710.json
outputs/riskkv_v19_v120_aggressive_fingerprint_pr_shortdecode_20260710_m20
```

v120 = v111 + PassageRetrieval 使用 v118 action，并通过 `--force_no_chat_tasks passage_retrieval_en` 保持该任务为 completion-style prompt。预期它会进一步降低 LongBench 总 KV keep，同时 score 接近 v111。
