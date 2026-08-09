# Section 161: learned router and same-sample checks

日期: 2026-07-11

## 1. BM25-bridge qasper 的 same-sample 修正

dashboard 的 partial table 使用任务级 full/v300 均值做比例, 对 M20 smoke 会有偏差。因此对 qasper BM25-bridge 做了 `(task, sample_id)` 对齐检查:

| Method | Samples | Score | Full same samples | v300 same samples | vs full | vs v300 | KV | Online |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v315 qasper b128 bm25 1536 | 20 | 0.5112 | 0.5255 | 0.5138 | 97.3% | 99.5% | 35.68% | 0.569s |
| v318 qasper b128 bm25 1280 | 20 | 0.4929 | 0.5255 | 0.5138 | 93.8% | 95.9% | 30.05% | 0.623s |
| v319 qasper b128 bm25 1024 | 20 | 0.4586 | 0.5255 | 0.5138 | 87.3% | 89.3% | 24.42% | 0.590s |

结论:

- 1536 budget 质量安全但 KV 高于 30%。
- 1280 budget KV 几乎到 30%, 但低于 95% full。
- 1024 budget 太激进。

因此 qasper BM25-bridge 不能作为单一默认动作, 只适合作为 router 的候选分支。

## 2. Strict oracle labels

用当前候选池生成了严格 same-sample oracle labels:

```text
outputs/riskkv_v19_oracle_action_labels_partial_strict_20260711/
```

覆盖 1600 个 full baseline 样本。标签定义为:

```text
safe action = score >= 0.95 * full_score_same_sample 且 KV <= 30%
```

这一步确认了固定预算不是合理主线: 很多任务只有部分样本能安全进入 <=30% KV。

## 3. Learned danger router v1

用 v300 的生成前特征训练了二分类 `danger` router:

输入特征包括:

- task family / metric / scorer;
- raw length / page count;
- keep fraction / budget;
- score max/mean/gap2/gap3/entropy;
- query coverage recall;
- score-risk / coverage-risk / verifier flags。

输出:

```text
danger vs safe_sparse
```

结果:

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| heuristic `score_risk_triggered` | 69.1% | - | F1=0.382 for danger |
| learned danger router | 73.1% | 0.712 | 0.735 |

learned router 明显比旧 heuristic 更会识别危险样本。

## 4. Learned policy simulation

但是把 learned danger router 直接用于“危险则 full fallback, 否则 v300”的策略, 离线交叉验证并不划算:

| Policy | Score | KV | Online | Fallback |
|---|---:|---:|---:|---:|
| full KV | 0.3692 | 100.00% | 3.0046s | 100% |
| v300 reference | 0.4426 | 27.64% | 0.5806s | 0% |
| danger fallback threshold 0.80 | 0.4425 | 30.45% | 0.9862s | 22.25% |
| danger fallback threshold 0.70 | 0.4412 | 32.77% | 1.2490s | 24.90% |
| action classifier policy | 0.4087 | 34.54% | 2.1232s | mixed |

结论:

- learned danger classifier 本身有信号;
- 但 full fallback 不是好动作, 因为 full KV 在部分样本上并不比 v300 分数更高, 且速度/KV 成本太大;
- 下一步 router 不应学“是否 full fallback”, 而应学“当前样本的最小安全稀疏动作”。

## 5. Short-decode partial check

v316 balanced short-decode 的 2wikimqa M20 已完成:

| Method | Score | vs full | vs v300 | KV | Online |
|---|---:|---:|---:|---:|---:|
| v316 2wikimqa | 0.3187 | 76.1% | 100.0% | 34.45% | 0.2247s |

该任务上 short-decode 与 v300 same-sample 分数完全一致, online 也几乎没有提升。因此 short-decode 暂时没有正信号, 需要等其它 QA 任务完成后再决定。

## 当前方向

主线保持 v300/v311 practical best。后续探索重点:

1. 不再优先扩展 B16/BM25 全任务实验。
2. 等 v316/v317 完成, 判断输出长度控制是否只对少数任务有效。
3. 训练/蒸馏 `min_safe_sparse_action` router, 而不是 full-fallback router。
4. 如果要加入 qasper BM25-bridge, 必须作为 conditional action, 不能全 qasper 默认使用。
