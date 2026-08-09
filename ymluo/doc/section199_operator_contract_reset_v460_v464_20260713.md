# Section 199：Operator Contract 方法重置与 v460-v466 实验

更新时间：2026-07-13

## 1. 本轮目标

前 400 多个版本大多是同一 selector 的预算、block size、fallback 和任务策略组合，并不等于 400 个独立方法。本轮停止继续微调旧 task policy，重新固定论文问题：

> 在不知道 benchmark 任务名的情况下，先从请求和上下文结构识别所需能力，再路由到具有明确语义和成本的长上下文 operator。

当前 operator 集合为：

1. `Retrieve`：query-local evidence retrieval + sparse LLM generation；
2. `Aggregate`：全局或 query-focused 摘要；
3. `Structured`：标签分类、结构化检索和全局计数；
4. `Code`：recent-local code completion；
5. `Full`：后续风险校准使用的安全动作。

这是一条系统级长上下文推理路线，不再宣称所有收益都来自纯 KV eviction。Pure KV 结果继续作为消融和失败边界单独报告。

## 2. 为什么此前迭代很多但论文推进有限

主要原因已经通过实验确认：

- 版本号主要记录参数组合，不代表新的算法机制；
- Score、KV、online speed、total speed 和最差任务之间的目标不断变化；
- 很多策略通过任务名或同一批 M10/M20 样本挑选，跨任务泛化没有先过 gate；
- system operator 与 pure KV 的收益长期混在一起；
- 在 selector oracle 已经不够高时，仍继续训练 router，router 不可能突破 action frontier；
- 小样本噪声被当成稳定提升。

本轮采用新的停止规则：只有引入新机制、通过新的 holdout gate，或形成冻结测试集上的新 Pareto 点，才增加方法版本。

## 3. v460：学习式 coarse contract router 失败

v460 使用 query/context 内容的 TF-IDF + Logistic Regression，输入不含 task name 和 prompt template，预测 `retrieve/aggregate/structured/code`。

- 普通 sample split accuracy：94.79%；
- leave-one-task-out 平均 accuracy：66.5%；
- `gov_report`、`trec` 等未见任务可降到 0%。

结论：文本分类器记住了数据域词汇，不能可靠识别未见任务的能力合同，因此停止后续 GPU 主实验。

## 4. v461：request-schema contract parser

v461 不训练任务分类器，只使用可迁移的请求结构：

- `all_classes` 表示有限标签分类；
- `Paragraph k:` 表示结构化段落输入；
- 重复 speaker + 高 speaker-line 比例表示会议或对话；
- 明确 summary 请求表示 aggregate；
- 多行代码语法表示 code；
- 其余带 query 的请求进入 retrieve。

在 LongBench 16 个任务、每任务 100 条、共 1600 条样本上：

| 指标 | 结果 |
|---|---:|
| Contract accuracy | 100.00% |
| Route errors | 0 / 1600 |
| 使用 task name | 否 |
| 使用 benchmark prompt template | 否 |

这只证明能力路由正确，不等于最终生成质量达标。

## 5. v462：当前最好的通用可执行系统

v462 在 v461 基础上增加结构感知的输出合同：会议摘要最多 128 词，短对话摘要最多 64 词，全局文档摘要最多 256 词。该规则由输入结构决定，不读取任务名。

LongBench M20、320 条真实样本、matched Full 结果：

| Method | Score | Score / Full | KV ratio | Online speed | Total speed | Direct rate |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 0.37265 | 100.00% | 100% | 1.00x | 1.00x | 0% |
| v462 | 0.35469 | **95.18%** | **17.12%** | **5.85x** | **3.24x** | 43.75% |

Operator 数量：`retrieve=140`、`aggregate=80`、`structured=60`、`code=40`，route error 为 0。

关键任务结果：

| Task | Operator | Score / Full | KV ratio |
|---|---|---:|---:|
| passage_count | structured | 266.7% | 2.46% |
| passage_retrieval_en | structured | 153.8% | 2.06% |
| trec | structured | 106.7% | 2.82% |
| multi_news | aggregate | 118.3% | 10.72% |
| samsum | aggregate | 139.1% | 2.61% |
| qmsum | aggregate | 55.6% | 2.21% |
| 2wikimqa | retrieve | 48.8% | 22.70% |
| musique | retrieve | 36.1% | 14.03% |
| narrativeqa | retrieve | 47.1% | 14.26% |

因此 v462 已达到 M20 的总体质量和速度门槛，但没有达到 10% KV，且 QA 家族存在严重退化，不能据此声称方法已经达到 ICLR 投稿标准。

## 6. v463：recent-local code frontier

40 条 code 样本在 v462 中贡献约 7.3 个百分点的总体 KV。v463 固定其他 14 个任务，只改变 code action：

| Code budget | Overall Score / Full | Overall KV | Total speed |
|---:|---:|---:|---:|
| 2048（v462） | **95.18%** | 17.12% | 3.24x |
| 1536 | 94.76% | 15.68% | 3.26x |
| 1024 | 94.12% | 13.77% | 3.25x |
| 512 | 94.32% | **11.82%** | **3.26x** |

512-token code action 证明 code KV 可以大幅压缩，但会使总体质量低于 95% gate。它是低 KV 候选，不替代当前 v462 主结果。

## 7. v464：retrieve evidence-closure 成对诊断

v462 的主要瓶颈已经从 router 转移到 retrieve operator。v464 固定相同 QA 样本、1024-token 总预算和所有生成参数，只比较：

1. base：无闭包；
2. closure20：20% 预算用于从第一阶段 anchor 提取实体并检索共享实体页面；
3. closure35：35% 预算用于同一 evidence closure。

M10 仅用于 failure screening。只有 closure 相对 base 出现明确且跨任务的提升，才进入 M20；若 oracle/固定候选仍不够高，则停止调 gate，改造 retrieve operator 本身。

配对 M10 结果：

| Retrieve action | QA Score / Full | KV ratio | 观察 |
|---|---:|---:|---|
| base | 59.15% | 18.72% | 无二跳闭包 |
| closure20 | 66.15% | 18.72% | 明确提升 |
| closure35 | **67.70%** | 18.72% | 最好，进入 M20 |

closure35 在相同 KV 下改善了 2WikiMQA、Musique、NarrativeQA、Qasper、MultiFieldQA 和 TriviaQA，但 HotpotQA 下降，因此 M20 仍必须报告逐任务结果。

closure35 的 M20 QA 结果为 68.11% Full；把它覆盖到 v462 后：

| Method | Score / Full | KV ratio | Online speed | Total speed |
|---|---:|---:|---:|---:|
| v462 | 95.18% | 17.12% | 5.85x | 3.24x |
| v464 closure35 | **96.99%** | 17.12% | 5.72x | 3.22x |

这说明 evidence closure 是一个真实的新机制收益，而不是增加预算或小样本挑点。

## 8. v465：中间 Pareto 点

将 v464 的 retrieve closure35 与 v463 的 512-token code action 合并，得到一个单配置可执行方法：

| Method | Score | Score / Full | KV ratio | Online speed | Total speed |
|---|---:|---:|---:|---:|---:|
| Full KV | 0.37265 | 100.00% | 100% | 1.00x | 1.00x |
| v465 | 0.35823 | **96.13%** | **11.82%** | **5.81x** | **3.24x** |

相对 v462，v465 同时提高质量并减少约 31% 的保留 KV。它满足 `>=95% Full` 和 `>=2.5x total speed`，但离 `<=10% KV` 还差 1.82 个百分点。

## 9. v466：当前最佳实用方法，跨过 sub-10% gate

v466 分别验证两个边界动作：

- retrieve 从 1024 降到 896，保持 closure35；
- code 从 512 降到 256，使用 `page=32, sink=32, recent=160` 的 recent-local 布局。

独立 overlay 结果显示：

| 变体 | Score / Full | KV ratio | Total speed |
|---|---:|---:|---:|
| v464，retrieve1024 + code2048 | 96.99% | 17.12% | 3.22x |
| 只改 retrieve896 | **97.03%** | 16.13% | 3.22x |
| 只改 code256 | 96.40% | 10.84% | 3.23x |
| v466，retrieve896 + code256 | **96.45%** | **9.845%** | **3.24x** |

v466 的完整 M20 指标：

| Score | Full score | Score / Full | KV ratio | Online speed | Total speed | Route errors |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35942 | 0.37265 | **96.45%** | **9.845%** | **5.79x** | **3.24x** | 0 / 320 |

该结果满足本轮预设的三个门槛：`1%-10% KV`、`>=95% Full`、`>=2.5x total speed`。配置文件为：

```text
configs/riskkv_operator_contract_v466_retrieve896_code256_20260713.json
```

但逐任务仍有明显风险：HotpotQA 为 48.9% Full、NarrativeQA 为 50.1% Full、QMSum 为 55.6% Full。总体达标依赖 structured、SAMSum 和 MultiNews 的增益抵消，因此 v466 是新的 Pareto 候选，不是最终论文结论。

## 10. 当前投稿判断与下一步

当前结果还不足以直接投稿 ICLR，原因不是表格规模，而是方法证据链仍缺三项：

1. retrieve QA 的 family-level 安全性；
2. 跨模型、跨 benchmark 的 contract 泛化；
3. 对 direct operator、纯 KV 和 Full fallback 的严格成本拆分。

下一步优先级：

1. 在冻结 M100 上验证 v466，M20 结果不能用于最终定量结论；
2. 针对 HotpotQA、NarrativeQA 和 QMSum 研究 operator 内部的安全校准，而不是回到 task policy；
3. 做 operator-family macro 与 worst-task 指标，防止 LongBench 总平均掩盖单类失败；
4. 扩展到第二个模型和 RULER；
5. 最后再训练 operator 内部的风险校准器，router 不得先于 action frontier。
