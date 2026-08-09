# section168：RiskKV-Block 面向 ICLR 的下一步主线：校准式动态预算规划器 v12

日期：2026-07-11

## 1. 当前判断

v350 在 M20 上看起来比 v300 更好，但 M100 复查后没有稳定超过 v300：

| 方法 | LongBench M100 score | KV keep | online seconds | 相对 full online |
|---|---:|---:|---:|---:|
| full KV | 0.3658 | 100.00% | 3.0988 | 1.00x |
| v300 robust action router | 0.4392 | 27.41% | 0.5632 | 5.50x |
| v350 clean budget router | 0.4339 | 26.70% | 0.5775 | 5.37x |

结论：v350 不是失败，但不能作为 ICLR 主方法。它证明了“小样本上学预算”有信号，但 M100 泛化不足。继续把 v350 包装成主结果风险很高。

## 2. 方法主线调整

下一步主线改成 **Risk-calibrated Dynamic KV Budget Planner**，核心不再是简单分类“该用哪个动作”，而是：

1. 对同一个样本枚举多个候选预算：B=256/384/512/768/1024/1536/2048/3072。
2. 用 M100 sweep 得到每个样本、每个预算的真实质量和代价曲线。
3. 训练一个 pairwise safety model：输入样本特征 + 候选预算特征，输出该预算是否安全。
4. 在 calibration fold 上选择安全概率阈值。
5. 推理时从小预算到大预算扫描，选择第一个超过校准阈值的预算；如果没有预算安全，则回退到 v293/v300 保守路径。

这比普通 router 更适合作为论文贡献，因为它有明确的风险约束、预算单调搜索、校准集阈值和 oracle 上界分析。

## 3. 已落实的工程

已新增 runtime 模式：

```text
ours_action_router_mode = learned_budget_planner_v2
```

它支持：

- 加载 `model.pkl`；
- 对多个 budget action 扫描安全概率；
- 用 metadata 里的 `safe_probability_threshold` 做校准；
- 输出最小安全预算；
- 未找到安全预算时回退；
- 可叠加 `v293_rules` 作为 base router，以保持和 v300/budget sweep 一致。

已新增训练脚本：

```text
scripts/train_calibrated_budget_planner_v12_m100_20260711.py
```

输出：

- `model.pkl`
- `metadata.json`
- `action_policy.json`
- `threshold_sweep.csv`
- `planner_predictions.csv`
- `planner_summary.csv`
- `feature_importance.csv`
- `outputs/riskkv_v19_budget_planner_v12_m100_compare_summary_20260711.csv`

已新增 watcher：

```text
scripts/watch_train_budget_planner_v11_v12_m100_20260711.sh
```

它会等待 8 个 M100 budget sweep 完成后，自动跑：

1. v11 clean budget router；
2. v12 calibrated budget planner；
3. 生成 `configs/riskkv_task_policy_v351_budget_planner_v12_m100_calibrated_20260711.json`；
4. 跑 v351 M20 smoke；
5. 若 M20 过 sanity gate，再跑 v351 M100。

## 4. 当前服务器状态

8 个 M100 budget sweep 正在跑：

```text
B=256,384,512,768,1024,1536,2048,3072
```

watcher 已后台启动。它不会抢当前 GPU，只会等 sweep 产出完整 `task_results.csv` 后接着训练。

## 5. 验收标准

如果 v12 要作为 ICLR 主方法，至少需要满足：

| 层级 | 指标 |
|---|---|
| LongBench M100 | score 不低于 v300，最好高于 v300；KV keep 明显低于 v300 |
| online speed | 不低于 v300；如果 KV 降得明显但 online 没兑现，需要解释瓶颈 |
| oracle gap | v12 距离 budget-sweep oracle 不能太远，否则说明 planner 学习不足 |
| 泛化 | 不能只在 M20 好看，必须 M100 稳定 |
| 论文故事 | 必须强调 calibrated minimum-safe budget planning，而不是手写任务规则 |

当前 v300 基线是：

```text
score = 0.4392
KV keep = 27.41%
online = 0.5632s
```

理想目标：

```text
score >= 0.439
KV keep <= 24-25%
online <= 0.54s
```

强结果目标：

```text
score >= 0.445
KV keep <= 22%
online speed vs full >= 5.8x
```

## 6. 如果 v12 不够好，下一步不是继续堆 router

若 v12 仍然不能稳定超过 v300，说明问题不在“预算预测”，而在某些任务的 retrieval/operator 本身质量上限不够。下一步应该转向：

1. 用 budget sweep 计算每个任务的 oracle 曲线，找出“给更多 budget 也救不回来”的任务。
2. 对这些任务做 retrieval 结构改造，而不是继续训练 router。
3. 优先检查 qasper、2wikimqa、musique、repobench-p、lcc，因为 v350 M100 的主要损失集中在这些任务。
4. 对 QA 类任务尝试 evidence bridge / entity graph / answer-type conditioned retrieval。
5. 对 code 类任务尝试 prefix/suffix structural packing，而不是普通文本 overlap。

这条路线更接近 ICLR 需要的“方法贡献”，而不是调参式 router。
