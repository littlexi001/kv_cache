# Section 167: Clean Budget Router v350 实际在线结果

日期：2026-07-11

## 当前结论

当前最好的实际可用 learned budget router 是：

- 配置：`configs/riskkv_task_policy_v350_clean_budget_router_v6_preserve_v293_20260711.json`
- 训练模型：`outputs/riskkv_v19_learned_budget_router_v6_noweight_covered320_preselection_both_conf050_20260711/model.pkl`
- 在线输出：`outputs/riskkv_v19_clean_budget_router_v6_preserve_v293_20260711_clean_budget_router_online_m20_m20_bDyn_pDyn`

这个版本的核心修正是：保留 v300 中原有的 `v293_rules` action router，不在 `narrativeqa` 和 `hotpotqa` 上用 learned router 覆盖原规则；其余任务使用 learned budget router 预测是否切到 `budget_b256` 或 `budget_b2048`，低置信度回退到 reference。

## 同样 320 个 LongBench 样本上的在线结果

| 方法 | Score | KV keep | Online seconds | Total seconds |
|---|---:|---:|---:|---:|
| full KV | 0.3727 | 100.00% | 3.0151 | 4.7094 |
| v300 手写 policy | 0.4461 | 25.95% | 0.4896 | 1.6702 |
| v350 learned budget router | 0.4476 | 24.60% | 0.4902 | 1.6608 |

相对 v300：

- Score：+0.00143，约 `+0.32%`
- KV keep：`25.95% -> 24.60%`，相对减少 `5.20%`
- Online speed：`0.999x`，基本持平
- Total speed：`1.006x`

相对 full KV：

- Score：`+20.10%`
- KV keep：`24.60%`
- Online speed：`6.15x`
- Total speed：`2.84x`

## 为什么 v349 不能当主结果

v349 把所有任务的 `action_router_mode` 都改成了 `learned_budget_v1`。这会破坏 v300 原本已经存在的 `v293_rules`，尤其是 `narrativeqa`。

具体问题：

- Offline replay 中的 `reference` 指向 v300 已跑出的结果，包含 v293 规则路由。
- Online v349 中的 `reference` 只是不做 learned budget override，但 v293 规则已经被 learned router 替换，因此不是同一个 reference。
- 这导致 `narrativeqa` 从 v300 的 `0.2538` 掉到 `0.1613`。

v350 修复方式：

- `narrativeqa` 和 `hotpotqa` 不启用 learned router，继续使用 v300 原本的 `v293_rules`。
- 其他原本 `action_router_mode=off` 的任务再启用 learned budget router。

## 本轮探索过但没有成为主方法的版本

| 版本 | 思路 | 结论 |
|---|---|---|
| v6 | clean preselection features，多类预算分类，去掉 class weight | 离线最好，接入 v350 后成为当前主方法 |
| v7 | budget-conditioned safety ladder，逐预算判断是否安全 | 太激进，held-out 质量不稳 |
| v8 | 只判断 B256 是否安全，不安全回 v300 | 可解释性强，但整体 score 不如 v6/v350 |
| v9 | 加回 query coverage features | held-out 有偶然高分点，但 all split 不稳，不作为主方法 |

## 当前 action 分布

v350 在线 M20 的 320 个样本：

| Action | Count |
|---|---:|
| reference | 206 |
| budget_b256 | 70 |
| v275_narrative_entropy | 18 |
| v291 | 12 |
| v287_hotpot_low_risk | 10 |
| budget_b2048 | 4 |

其中 `v275_narrative_entropy`、`v291`、`v287_hotpot_low_risk` 来自保留的 v293 规则；`budget_b256` 和 `budget_b2048` 来自 learned budget router。

## 下一步

1. 把 v350 跑到 M100，验证 `+0.32% score / -5.2% KV` 是否在更大样本上稳定。
2. 如果 M100 稳定，再跑 Table 5 同配置的完整 LongBench question-aware 对比。
3. 训练数据不能只用 M20 budget sweep。下一步应扩到 M100 budget sweep，至少覆盖 `B256/B384/B512/B1024/B2048`，否则 router 的泛化上限太低。
4. 论文叙事上不要说“端到端大幅加速 v300”，当前更准确的说法是：learned router 自动替代部分手写预算选择，在保持 v300 质量的同时进一步降低 KV；相对 full KV 的在线加速仍然很强。
