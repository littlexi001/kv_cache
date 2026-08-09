# Section 196：v437 严格泛化验证与 RiskKV-Pure 实验协议

更新时间：2026-07-12

## 1. 本轮目标

本轮不继续搜索新参数，优先验证当前最好结果是否满足 ICLR 论文需要的可信度：

1. 用真实端到端运行验证 v437，而不是只使用离线 policy simulation；
2. 用 Leave-One-Task-Out（LOTO）和 Leave-One-Family-Out（LOFO）检查 source-router 是否依赖 LongBench 任务模板；
3. 构建 `RiskKV-Pure`，强制所有样本经过标准 LLM prefill/decode，分离 KV 压缩收益与 direct operator 收益；
4. 所有实验同时报告平均分、KV ratio、online latency、最差任务与额外开销。

## 2. 已确认的 v437 M20 真实结果

v437 已完成 LongBench M20 真实运行，共 16 个任务、320 个样本：

| 指标 | 结果 |
|---|---:|
| Score | 0.3959 |
| KV ratio | 5.94% |
| Mean online | 0.4004 s |
| Mean query | 0.0471 s |
| Mean KV gather | 0.0030 s |
| Mean decode | 0.3502 s |

这说明 v437 的 runtime source-router 可以真实执行，但该结果仍包含 direct operator，不能直接作为纯 KV compression 主表结果。

## 3. Direct operator 审计

对 v437 M20 的逐样本结果统计：

| 指标 | 结果 |
|---|---:|
| Direct samples | 134 / 320 |
| Direct 使用率 | 41.88% |
| Direct 样本平均分 | 0.4263 |
| 标准生成样本平均分 | 0.3740 |
| Direct 样本 mean online | 0.0420 s |
| 标准生成样本 mean online | 0.6586 s |

使用 direct operator 的任务包括：

- `gov_report`：20/20；
- `qmsum`：20/20；
- `multi_news`：20/20；
- `trec`：20/20；
- `samsum`：14/20；
- `passage_count`：20/20；
- `passage_retrieval_en`：20/20。

因此，v437 当前的整体速度由两部分共同构成：低 KV 标准生成加速，以及绕过标准 prefill/decode 的 direct operator 加速。论文中必须分别报告。

## 4. v438 严格留出协议

v438 对 source-router 做两组交叉验证：

### 4.1 Leave-One-Task-Out

每轮完整移除一个 LongBench 任务。训练和阈值校准均不能使用该任务的任何样本，最后在该任务全部 M100 样本上评估。

### 4.2 Leave-One-Family-Out

任务族定义如下：

| Family | Tasks |
|---|---|
| Single-doc QA | narrativeqa, qasper, multifieldqa_en |
| Multi-doc QA | hotpotqa, 2wikimqa, musique |
| Summarization | gov_report, qmsum, multi_news |
| Few-shot | trec, triviaqa, samsum |
| Synthetic | passage_count, passage_retrieval_en |
| Code | lcc, repobench-p |

每轮完整移除一个任务族，检查 router 能否向未见任务族迁移。

### 4.3 模板泄漏对照

每个 holdout 同时测试两种 router 输入：

- `full_prompt`：prefix template、suffix template、query、context；
- `content_only`：只保留 query 与 context，移除固定任务模板。

如果 `full_prompt` 明显优于 `content_only`，说明 router 可能主要依赖 benchmark 模板识别，而不是学习请求级风险。

## 5. v439 RiskKV-Pure 定义

v439 继承 v437 的 source-router、block scorer、动态 budget 和 fallback，但在 source policy 合并完成后强制执行：

```json
{
  "__runtime_constraints": {
    "direct_structured_answer": false
  }
}
```

该约束在所有 task/source overlay 之后生效，因此任何子策略都不能重新打开 direct answer。验收条件：

| Gate | 条件 |
|---|---:|
| Direct usage | 0 |
| M20 Score | >= 0.30 |
| KV ratio | <= 12% |
| M100 quality retention | >= 95% Full |
| M100 actual speed | >= 2.5x Full |

## 6. 当前后台任务

| 实验 | 资源 | 状态 |
|---|---|---|
| v437 LongBench M100 real runtime | GPU 0 | 运行中 |
| v438 LOTO/LOFO + template ablation | CPU | 运行中 |
| v439 RiskKV-Pure M20 gate | GPU 6 | 运行中；通过后自动进入 M100 |

## 7. 结果解释规则

1. v437 强、v439 也强：可以把纯 KV 方法作为论文主线，direct operator 作为系统增强；
2. v437 强、v439 明显下降：论文主线应定位为 query-aware adaptive long-context inference system，并同时比较 RAG/retrieval 与 KV 方法；
3. LOTO 强、LOFO 弱：router 能适应同类新任务，但不能宣称跨任务族通用；
4. `full_prompt` 强、`content_only` 弱：需要重新训练去模板 router，并加入结构化、retrieval-stability 和风险特征；
5. Musique 等最差任务明显退化：优先增加风险 fallback，不使用平均分掩盖失败。

## 8. 完成结果（2026-07-12 更新）

### 8.1 v437 真实 M100

| 指标 | Full | v437 | 相对结果 |
|---|---:|---:|---:|
| Score | 0.3658 | 0.3807 | 104.06% |
| KV ratio | 100% | 6.99% | 14.30x KV reduction |
| Mean online | 3.0988 s | 0.4530 s | 6.84x |
| Mean total | 4.7802 s | 1.4947 s | 3.20x |

v437 的真实 M100 与离线预测完全一致，说明 source-router runtime 没有造成模拟到真实执行的落差。但是 669/1600（41.81%）样本使用 direct operator，因此该结果应作为系统版本，而不是纯 KV compression 版本。

### 8.2 v438 严格留出

| Router input | Holdout | Score / Full | KV | Speed（离线 online） | Source label acc. |
|---|---|---:|---:|---:|---:|
| Full prompt | Task | 102.97% | 6.66% | 6.63x | 72.63% |
| Content only | Task | 104.08% | 6.70% | 6.34x | 71.63% |
| Full prompt | Family | 100.40% | 4.95% | 8.11x | 50.00% |
| Content only | Family | 102.55% | 6.60% | 6.81x | 53.06% |

去掉固定 prompt 模板后总体结果没有下降，因此没有发现明显的模板记忆证据。但 family holdout 的 source label accuracy 只有约 50%，总体质量仍高主要依赖 fallback 到强 base policy，不能据此宣称 source-router 已经跨任务族泛化。

最差 family holdout：

| Family | Score / Full（content only） | 结论 |
|---|---:|---|
| Single-doc QA | 60.53% | 严重失败 |
| Multi-doc QA | 62.85% | 严重失败 |
| Summarization | 88.37% | 未达到 95% gate |
| Code | 98.51% | 通过质量 gate，但 KV 为 18.62% |

最差 task holdout 仍是 `musique`（21.20% Full）、`narrativeqa`（60.25%）和 `qmsum`（67.58%）。下一版 router 必须训练样本级风险与安全动作，不能仅预测 source policy。

### 8.3 v439 Pure M20

匹配 320 个样本：

| 指标 | Full | v439 | 相对结果 |
|---|---:|---:|---:|
| Score | 0.3727 | 0.2652 | 71.16% |
| KV ratio | 100% | 5.94% | 16.84x KV reduction |
| Mean online | 3.0151 s | 2.0316 s | 1.48x |
| Mean total | 4.7094 s | 3.7551 s | 1.25x |

v439 未通过质量和速度 gate，并且 Samsum 的 action-router 在后置阶段重新打开了 14 次 direct answer。因此已增加 post-route runtime constraint，并启动 v440 真正 Pure M20。v439 已足以证明：当前 v437 的高质量与高速度不能全部归因于 KV block compression。

### 8.4 当前论文定位

当前证据支持把 v437 定位为 `query-aware adaptive long-context inference system`：它联合选择 KV frontier、retrieval/generation operator 和 fallback，并在 LongBench 上实现 6.99% KV、104.06% Full quality、3.20x 真实 total speed。

当前证据不支持把 v437 直接表述为纯 KV cache compression SOTA。纯 KV 主线是否可行，需要以 v440 和后续重新训练的 risk-aware pure router 为准。
