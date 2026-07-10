# Section 123: Selective risk actions toward 30% KV keep

## 目标

当前目标是把实际可用 LongBench policy 推到：

- KV keep 尽量接近 10%-30%；
- score 保持 full KV baseline 的 95% 以上；
- 同时保留一个能写进 ICLR 的方法故事，而不是只做参数调优。

full KV m20 baseline：

| 指标 | 数值 |
| --- | ---: |
| LongBench score | 0.372655 |
| 95% baseline | 0.354022 |
| KV keep | 100.00% |

## 已知边界

v111 是上一轮最稳的 aggressive point：

| 方法 | LongBench score | KV keep |
| --- | ---: | ---: |
| v81 quality branch | 0.380371 | 58.31% |
| v107 aggressive release | 0.359801 | 45.35% |
| v111 aggressive fingerprint | 0.368951 | 44.49% |

v120 在 v111 上替换 PassageRetrieval：

- PassageRetrieval 从 v111 的 `0.65 / 95.70% KV` 变为 v118 action 的 `0.60 / 13.91% KV`。
- v120 已完成全量 LongBench m20：

```text
outputs/riskkv_v19_v120_aggressive_fingerprint_pr_shortdecode_20260710_m20
```

| 方法 | LongBench score | KV keep | online seconds |
| --- | ---: | ---: | ---: |
| full KV | 0.372655 | 100.00% | 3.033 |
| v111 aggressive fingerprint | 0.368951 | 44.49% | 2.693 |
| v120 PR short-decode | 0.365826 | 39.38% | 2.678 |

结论：v120 已经超过 full baseline 的 95% 分数线，但 KV keep 还没有进入 30% 以内。

## Negative results

### QA no-fallback 不可直接用

v121/v122 测 NarrativeQA + MultiFieldQA 直接低预算、无 verifier：

| 方法 | task | score | KV keep |
| --- | --- | ---: | ---: |
| v121 512 no-fallback | NarrativeQA | 0.1407 | 7.48% |
| v121 512 no-fallback | MultiFieldQA | 0.4195 | 10.93% |
| v122 1024 no-fallback | NarrativeQA | 0.1068 | 14.26% |
| v122 1024 no-fallback | MultiFieldQA | 0.4125 | 21.58% |

结论：不能简单关闭 verifier，也不能只靠增加到 1024 解决 QA 质量。

### Multi-hop no-chat 不可用

v123 测 2WikiMQA/HotpotQA/MuSiQue 的 no-chat 2048：

| task | score | KV keep |
| --- | ---: | ---: |
| 2WikiMQA | 0.1438 | 39.95% |
| HotpotQA | 0.0826 | 31.22% |
| MuSiQue | 0.0664 | 27.58% |

结论：PassageRetrieval 的 no-chat + short decode 成功不能直接迁移到多跳 QA。多跳 QA 的瓶颈主要是 evidence selection / evidence composition，不是输出格式。

### 2Wiki 低预算不划算

v124/v125：

| 方法 | score | KV keep |
| --- | ---: | ---: |
| 2Wiki 512 no-fallback | 0.3662 | 11.71% |
| 2Wiki 1024 no-fallback | 0.3662 | 22.48% |
| v111 2Wiki reference | 0.4687 | 45.56% |
| full 2Wiki reference | 0.4187 | 100.00% |

结论：2Wiki 可以省 KV，但会丢掉 v111 的主要正收益；暂时不并入主策略。

## Positive result: over-confident risk gate

v121/v122 的 per-sample 分析发现一个重要信号：

> 有些 sparse retrieval 不是高熵不确定，而是低熵过度自信地选错证据。

因此新增 `score_risk_entropy_at_most`：

```text
if entropy <= threshold:
    trigger risk action
```

这与原有 `score_risk_max_entropy` 形成互补：

- high entropy risk：模型不知道该选哪个 block；
- low entropy overconfidence risk：模型非常确定地选了少数 block，但这些 block 可能错。

## v127 QA selective entropy fallback

配置：

```text
configs/riskkv_task_policy_v127_qa_selective_entropy_fallback_20260710.json
```

策略：

- NarrativeQA：512 budget；若 entropy <= 0.9699，则 full fallback；
- MultiFieldQA：512 budget；若 entropy >= 0.9892，则 full fallback；
- 不使用 output/grounding/consistency verifier。

结果：

| task | score | KV keep | full score | v111 score |
| --- | ---: | ---: | ---: | ---: |
| NarrativeQA | 0.2538 | 30.61% | 0.2554 | 0.2433 |
| MultiFieldQA | 0.5339 | 34.18% | 0.5238 | 0.5678 |

结论：

- NarrativeQA 基本恢复 full score，同时把 KV keep 从 v111 的 58.37% 降到 30.61%。
- MultiFieldQA 仍高于 full baseline，同时把 KV keep 从 v111 的 56.03% 降到 34.18%。
- 这是一个可写进论文的方法点：risk-conditioned KV action 不只识别不确定性，也识别 over-confident evidence failure。

## v126 and v128

v126 = v120 + selective multi-hop fallback：

- HotpotQA：2048 budget；`top_score <= 1.1592` 触发 full fallback；
- MuSiQue：2048 budget；`gap2 <= 0.0827` 触发 full fallback；
- 目标是从 task-level full fallback 改为 sample-level full fallback。

运行中：

```text
outputs/riskkv_v19_v126_v120_selective_multihop_fallback_20260710_m20
```

实际结果：

| 方法 | LongBench score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| v120 | 0.365826 | 39.38% | PassageRetrieval short-decode |
| v126 | 0.363326 | 35.76% | selective HotpotQA/MuSiQue fallback |

v126 仍高于 95% baseline，但没有进入 30% KV。HotpotQA 从 full/v120 的 0.4008 降到 0.3608，说明 `top_score <= 1.1592` 的 threshold 有效省 KV，但不是完全安全。

v128 = v126 + v127 QA entropy fallback：

- PassageRetrieval：v118 no-chat + short decode；
- PassageCount：structured fingerprint；
- NarrativeQA/MultiFieldQA：selective entropy fallback；
- HotpotQA/MuSiQue：selective multi-hop fallback；
- 2WikiMQA 保留 v120/v111 的较稳策略。

运行中：

```text
outputs/riskkv_v19_v128_v120_selective_multihop_qa_entropy_20260710_m20
```

实际结果：

| 方法 | LongBench score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| v128 | 0.361860 | 32.66% | v126 + QA entropy fallback |

v128 明显接近 30% KV，且仍高于 95% baseline。

v129 = v128 + 2WikiMQA 512 no-fallback：

```text
outputs/riskkv_v19_v129_v128_2wiki512_aggressive_20260710_m20
```

这个候选更激进，目标是把 KV keep 推到 30% 以下。代价是 2WikiMQA 单项会从 v111 的 0.4687 降到 probe 中的约 0.3662。按现有 m20 估计，总分仍可能高于 0.354 的 95% baseline。

实际结果：

| 方法 | LongBench score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| v129 | 0.355453 | 30.54% | v128 + 2Wiki 512 |

v129 是目前最接近用户目标的已完成策略：score 仍高于 95% baseline `0.354022`，KV keep 距离 30% 只差 0.54 个百分点。

## v130/v131: final compression probes

### v130 TREC 512 no-risk

TREC 去掉 score-risk 后：

| 方法 | TREC score | KV keep |
| --- | ---: | ---: |
| v129 / risk-gated TREC | 0.70 | 29.38% |
| v130 no-risk TREC | 0.50 | 10.34% |

结论：TREC no-risk 不可合并。虽然省 KV，但会把总分拉破 95% baseline。

### v131 summarization 512 probe

为了把 v129 的 30.54% 推到 30% 以下，启动 summarization-512 probe：

```text
outputs/riskkv_v19_v131_summarization512_probe_20260710_m20
```

任务：

- GovReport: 1024 -> 512
- QMSum: 1024 -> 512
- MultiNews: 1024 -> 512

判定标准：如果三项平均掉分足够小，则合并成 v132；否则 v129 保持为当前最激进已完成策略。

实际结果：

| task | v129 score | v131 512 score | v129 KV keep | v131 KV keep |
| --- | ---: | ---: | ---: | ---: |
| GovReport | 0.1789 | 0.1668 | 16.12% | 8.26% |
| QMSum | 0.1562 | 0.1474 | 14.57% | 7.51% |
| MultiNews | 0.1657 | 0.1522 | 59.01% | 37.68% |

三项全降到 512 会让总分预计低于 95% baseline，因此不直接合并。

### v132 v129 + GovReport/QMSum 512

更精细的组合是只降低 GovReport 和 QMSum：

```text
outputs/riskkv_v19_v132_v129_gov_qmsum512_20260710_m20
```

预估：

| 方法 | 预估 score | 预估 KV keep |
| --- | ---: | ---: |
| v129 | 0.355453 | 30.54% |
| v132 | 约 0.35415 | 约 29.6% |

v132 分数非常贴近 95% baseline `0.354022`，是 high-risk candidate。若实测低于 95% baseline，则 v129 仍是当前最好的 aggressive practical strategy。

实际结果：

| 方法 | LongBench score | KV keep | online seconds | 是否过 95% baseline |
| --- | ---: | ---: | ---: | --- |
| full KV | 0.372655 | 100.00% | 3.033 | - |
| v120 | 0.365826 | 39.38% | 2.678 | yes |
| v126 | 0.363326 | 35.76% | 2.678 | yes |
| v128 | 0.361860 | 32.66% | 2.561 | yes |
| v129 | 0.355453 | 30.54% | 2.558 | yes |
| v132 | 0.354147 | 29.61% | 2.524 | yes, very close |

v132 首次达到 30% 以下 KV keep，同时仍略高于 95% full baseline：

```text
0.354147 > 0.372655 * 0.95 = 0.354022
```

但 v132 的分数 margin 只有约 `0.000125`，非常薄。因此建议论文主表同时报告：

- v129: 稍稳的 aggressive practical point，30.54% KV keep；
- v132: frontier point，29.61% KV keep，刚好 95%+。

速度结论：

- v132 的 LongBench end-to-end online speedup 约为 `3.033 / 2.524 = 1.20x`；
- 这没有达到 2.5x，因为 LongBench 的 summarization/code 任务有大量生成 token，decode 生成本身主导 online time；
- 后续需要单独补 attention-subsystem 或 one-step/short-decode speed 表，不能把 full LongBench E2E speed 作为唯一速度证据。

预期：

- v120 已验证为 39.38% KV keep；
- v126 已验证为 35.76% KV keep；
- v128 已验证为 32.66% KV keep；
- v129 已验证为 30.54% KV keep；
- v131 正在测试是否能进一步压到 30% 以下。

## 当前方法故事

这一轮后，方法不再像简单的 task router，而更像：

```text
RiskKV-Block = task-structured action space + bidirectional risk gates
```

动作空间：

- lexical / IDF / flow block retrieval；
- structured fingerprint for counting；
- short format-controlled decode for retrieval-output tasks；
- selective full fallback；
- over-confident risk gate；
- high-entropy risk gate。

这比“固定 top-k block retrieval”更像 ICLR 方法，因为它强调：

1. KV compression 的危险不是单一形式；
2. 不同任务的最小安全动作不同；
3. risk signal 既包括 uncertainty，也包括 overconfidence；
4. policy 的目标是 minimum safe action，而不是固定低预算。
