# section169：Candidate-conditioned planner v13/v14 进展与下一步

日期：2026-07-12

## 1. 结论先行

这轮实验说明：继续只训练“样本级 router”已经很难吃到 oracle gap，真正有效的方向是 **candidate-conditioned minimum-safe budget planning**。也就是不只问“这个样本危险吗”，而是对每个候选动作都构造一组候选特征，预测“这个候选动作在当前样本上是否安全”，再选择最小安全动作。

当前最稳的 practical 点：

| 方法 | LongBench M100 score ratio vs v300 | KV relative vs v300 | 绝对 KV keep | speed vs v300 | fallback |
|---|---:|---:|---:|---:|---:|
| v13 family-calibrated | 1.0054 | 94.14% | 25.81% | 1.009x | 31.38% |
| v14 task-calibrated strict | 1.0009 | 90.66% | 24.85% | 1.059x | 31.25% |
| v14 task-calibrated 0.995-risk | 0.9969 | 89.89% | 24.64% | 1.057x | 31.25% |
| v14 task-calibrated 0.990-risk | 0.9908 | 89.20% | 24.45% | 1.064x | 31.25% |

这里的 v300 reference 为：

```text
score = 0.439235
KV keep = 27.41%
```

因此 v14 strict 已经把绝对 KV keep 从 27.41% 降到 24.85%，同时整体分数略高于 v300。这个点比 v12 有意义得多，也是目前最接近 oracle gap 的 practical 结果。

## 2. v12 为什么失败

v12 是 calibrated budget planner，但只看样本特征和预算动作本身，没有看“候选动作实际会选中哪些 block”。结果校准后过于保守：

```text
score ratio = 1.0
KV relative = 1.0
fallback rate = 1.0
```

这说明问题不是随机森林不够强，而是输入信息不够。对于 KV block retrieval，同一个 B=512 在不同样本、不同 query coverage、不同 top-block gap 下风险完全不同。只给 budget id，不给候选动作的检索形态，模型没法判断“这个候选动作为什么安全”。

## 3. v13/v14 的方法变化

v13/v14 的核心是把 planner 从 sample-conditioned 改成 candidate-conditioned：

1. 对每个样本枚举多个候选动作，例如不同 block size、不同 top-k、不同 budget。
2. 对每个候选动作读取 sweep 中真实产生的候选统计：
   - `candidate_keep_fraction`
   - `candidate_selected_pages`
   - `candidate_query_coverage`
   - `candidate_score_max/mean/gap/entropy`
   - `candidate_kv_relative_to_reference`
   - `candidate_kept_context_relative_to_reference`
   - `delta_selected_density_vs_reference`
3. 训练二分类安全模型，标签为：
   ```text
   candidate_score >= quality_ratio * reference_score
   ```
4. 推理时从小 KV 候选动作开始扫，选择第一个安全概率超过阈值的动作。
5. 如果没有动作安全，则 fallback 到 v300 reference。

v13 使用 family-level calibration；v14 改成 task-level calibration。v14 明显更好，因为 LongBench 各任务的风险分布差异很大，按 family 校准会浪费很多可压缩空间。

## 4. 当前最佳点

### 4.1 严格质量优先

推荐作为当前主结果的点：

```text
riskkv_v19_candidate_conditioned_planner_v14_taskcal_q10_rf_none_d0_l4_cal095_20260712
```

整体结果：

| 指标 | 数值 |
|---|---:|
| samples | 1600 |
| score ratio vs v300 | 1.0009 |
| learned score | 0.439621 |
| v300 reference score | 0.439235 |
| KV relative vs v300 | 90.66% |
| absolute KV keep | 24.85% |
| speed vs v300 | 1.059x |
| safe rate | 95.06% |
| fallback rate | 31.25% |

按任务看，主要收益来自：

| task | score ratio | KV relative vs v300 | 说明 |
|---|---:|---:|---|
| qmsum | 1.0721 | 72.01% | v14 task calibration 吃到了 v13 没吃到的空间 |
| triviaqa | 1.0343 | 52.59% | 非常适合短答案/低预算动作 |
| lcc | 0.9959 | 65.37% | 可显著压缩，质量基本守住 |
| 2wikimqa | 1.0005 | 87.53% | 稳定小幅收益 |
| qasper | 0.9622 | 90.79% | 质量有损失，是当前主要风险之一 |
| repobench-p | 0.9865 | 88.11% | 有压缩，但仍需代码结构感知改进 |

被 fallback 或基本不再压缩的任务：

| task | 原因 |
|---|---|
| gov_report | v300 已经只保留约 2.47% KV，再压没有意义，oracle 甚至可能更大 |
| multi_news | v300 已经约 9.66% KV，继续压缩收益有限且摘要风险高 |
| trec / passage_count / passage_retrieval | v300 已经极低 KV，动态 planner 通常 fallback 是合理的 |

### 4.2 轻微风险但更低 KV

如果允许整体分数约 0.3% 的损失，当前更激进点为：

```text
riskkv_v19_candidate_conditioned_planner_v14_taskcal_q099_rf_none_d0_l4_cal094_20260712
```

结果：

| 指标 | 数值 |
|---|---:|
| score ratio vs v300 | 0.9969 |
| KV relative vs v300 | 89.89% |
| absolute KV keep | 24.64% |
| speed vs v300 | 1.057x |

这个点不适合作为主结果，但可以放在 Pareto curve 上，说明质量-压缩可调。

## 5. 为什么还没有达到 oracle gap

当前 oracle 是在候选动作集合里事后挑最优动作，它的上界本身并不代表“能无限压缩”。从已有结果看：

```text
v14 strict: KV relative vs v300 = 90.66%
oracle:     KV relative vs v300 = 92.54% 左右
```

注意这里 oracle 的定义偏向“满足质量的最小候选动作”，但不同 quality ratio 下会有波动，有些任务的 oracle 甚至比当前 planner 更高 KV，因为它更重质量。真正能吃的空间主要来自几个任务，而不是所有任务：

- `qmsum`：task calibration 后能明显下降。
- `triviaqa`：短答案任务能极低预算。
- `lcc`：能大幅压缩，但代码结构风险还没完全建模。
- `qasper / multifieldqa / repobench-p`：存在质量风险，是继续拉近 oracle gap 的主要瓶颈。

因此下一步不是单纯调阈值，而是把 candidate features 做到 runtime，并增强候选动作的结构表达。

## 6. 论文故事怎么讲

现在比较有希望的故事是：

> KV compression should not be a fixed-budget or sample-only routing problem. The safe budget depends on the evidence geometry induced by each candidate compression action. We therefore formulate dynamic KV compression as candidate-conditioned calibrated minimum-safe action selection.

可以拆成三个贡献：

1. **Candidate-conditioned safety planning**  
   不只根据 query/task/sample 预测预算，而是根据候选动作的 block evidence geometry 预测安全性。

2. **Calibrated minimum-safe action selection**  
   用 calibration fold 学每个任务的安全阈值，从小预算到大预算选择最小安全动作；找不到安全动作时 fallback。

3. **Evidence-geometry features for KV block retrieval**  
   让 planner 看见候选动作的 selected block density、coverage、score gap、entropy、KV saving 等结构信息，而不是只看 budget id。

这个故事比“又训练了一个 router”强很多，也更容易和 AdaKV/Pyramid/SnapKV 区分开：我们的中心不是固定预算下按 attention/head 分配容量，而是在 question-aware block retrieval 上做风险校准和候选动作选择。

## 7. 现在最重要的下一步

v13/v14 目前是 offline planner：候选动作特征来自已跑完的 sweep。要让它变成真正 practical method，需要做 runtime 化：

1. 在 `keep_ours_page` 里把 block scoring 和 block selection 拆开。
2. 对候选动作做 dry-run，不真实构造 KV，只返回：
   - candidate selected block ids
   - candidate coverage
   - candidate score stats
   - candidate KV estimate
3. 把这些 dry-run features 喂给 v14 model。
4. 选择最小安全动作后，再只执行一次真实 KV gather。
5. 做 M20 smoke，确认 runtime planner 的输出动作和 offline planner 接近。
6. 再跑 M100，比较：
   - offline v14
   - runtime v14 approximate
   - v300

如果 runtime 误差很小，这条线就可以作为论文主方法继续推进。如果 runtime 误差很大，就需要把 selection dry-run 做成和真实 selection 完全一致。

## 8. 当前判断

当前结果还不能说“已经足够发 ICLR”，因为 v14 仍是 offline planner，且主结果只是在 v300 基础上进一步降低约 2.56 个绝对 KV 点。但这是最近几轮里最有价值的正向信号：

```text
v300:       score 0.4392, KV 27.41%
v14 strict: score 0.4396, KV 24.85%
```

这说明方法不是靠牺牲质量换速度，而是在 candidate-conditioned risk modeling 后确实找到了更低的安全预算。下一步只要 runtime 化成功，并在更完整 LongBench / RULER / 多模型上复现，这条线可以作为 ICLR 主线继续写。
