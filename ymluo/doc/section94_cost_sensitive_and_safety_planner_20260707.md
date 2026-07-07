# Section 94: Cost-sensitive / safety planner 初试（2026-07-07）

## 本轮目标

上一节已经把 variable-budget planner 接入真实 runtime，但暴露出两个问题：

1. best-label checkpoint 在 runtime 上很省 KV，但概率过度尖锐，常常过度选择 `k1_compact`。
2. LongBench worst-case 上，低预算能达到 full-level，但没有超过 prompt rebuild / oracle-with-full。

本轮尝试两条路线：

- cost-sensitive multiclass loss；
- action-level safety verifier。

结论先行：

**这两条路线的首版都不能替代当前主方法。当前主线仍然是 runtime variable-budget planner；min-safe checkpoint 是更安全但更保守的 fallback。**

## Cost-sensitive multiclass loss

改动文件：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_variable_budget_planner_from_repack_results.py`

新增训练参数：

- `--ce_loss_weight`
- `--expected_cost_weight`
- `--unsafe_cost_weight`
- `--best_gap_cost_weight`
- `--kv_cost_weight`
- `--include_full_action`

每个 action 的 cost：

`unsafe_cost_weight * max(0, full_score - action_score)`

`+ best_gap_cost_weight * max(0, best_score - action_score)`

`+ kv_cost_weight * kv_ratio`

训练目标是在 CE 之外最小化预测分布下的 expected cost。

实验目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_cost_sensitive_grid_qwen8b_m4_plus_longbench12_20260707`

测试配置：

| name | CE | expected cost | unsafe | best gap | KV |
|---|---:|---:|---:|---:|---:|
| ce1_cost1_u4_b2_kv005 | 1.0 | 1.0 | 4.0 | 2.0 | 0.05 |
| ce03_cost1_u4_b2_kv01 | 0.3 | 1.0 | 4.0 | 2.0 | 0.10 |
| ce0_cost1_u4_b2_kv005 | 0.0 | 1.0 | 4.0 | 2.0 | 0.05 |
| ce05_cost1_u8_b2_kv005 | 0.5 | 1.0 | 8.0 | 2.0 | 0.05 |
| ce05_cost2_u8_b3_kv002 | 0.5 | 2.0 | 8.0 | 3.0 | 0.02 |

结果概要：

- learned planner 多数在 test 上只有约 28.57% score；
- full / oracle 在对应 split 上约 31.43%；
- risk-tail 若要恢复 full-level，通常需要很高 KV。

结论：

简单 expected-cost + CE 不是有效解。它会让模型更保守或更混乱，但没有可靠提升质量/KV tradeoff。下一版需要更像 constrained optimization，而不是把几个 cost 线性加到 CE 上。

## Post-hoc temperature

改动文件：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py`

新增 runtime 参数：

- `--variable_budget_temperature`

实验对象：

- 原始 best-label checkpoint；
- LongBench m=2；
- 强制 source=`group1__k1__k2__k3__k4__k6__k8`。

结果：

| setting | score | KV ratio |
|---|---:|---:|
| temperature=1 | 20.00% | 16.25% |
| temperature=10 | 10.00% | 41.25% |
| temperature=50 | 10.00% | 50.00% |

结论：

温度能让预算升高，但没有提升质量，反而掉分。当前 checkpoint 的问题不是单纯 softmax 温度，而是风险标签/模型本身没有学到可靠的 worst-case 升级边界。

## Min-safe checkpoint runtime

实验目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_variable_budget_qwen8b_longbench_m2_group1_minsafe_20260707`

结果：

| method | score | KV ratio |
|---|---:|---:|
| full KV cache | 20.00% | 100.00% |
| prompt rebuild | 30.00% | 26.39% |
| best-label variable-budget | 20.00% | 16.25% |
| min-safe variable-budget | 20.00% | 33.75% |

动作分布：

- `k1_compact`: 6/10
- `k2_compact`: 1/10
- `k3_compact`: 1/10
- `k8_compact`: 2/10

结论：

min-safe checkpoint 会主动升预算，确实比 best-label 更保守；但它仍只达到 full-level，没有追上 prompt rebuild / oracle-with-full。它可以作为安全 baseline，但不是最终主方法。

## Action-level safety verifier

新增脚本：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_action_safety_budget_planner_from_repack_results.py`

方法：

把每个 case 展开为多个 action-level 样本：

- 输入：case features + candidate budget/action features；
- 标签：该 action 是否 `score >= full_score`；
- 推理：按预算从小到大找第一个 `p(safe) >= threshold` 的 action，否则 fallback full。

单 split 结果：

目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/action_safety_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_20260707`

| policy | score | KV ratio |
|---|---:|---:|
| fixed full | 30.56% | 100.00% |
| safety verifier | 30.56% | 24.60% |
| oracle min-safe | 30.56% | 14.19% |
| oracle best | 36.11% | 16.62% |

多 seed 结果：

目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/action_safety_multiseed_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_20260707`

| policy | score mean | KV mean | full-level runs |
|---|---:|---:|---:|
| fixed full / k8 | 29.03% | 100.00% | 5/5 |
| oracle min-safe | 29.55% | 13.84% | 5/5 |
| oracle best | 32.57% | 15.26% | 5/5 |
| safety verifier tau=0.5 | 27.77% | 18.06% | 2/5 |
| safety verifier tau=0.95 | 27.77% | 19.71% | 2/5 |

结论：

action-level safety verifier 的想法有论文方法感，但当前数据量/校准不够，稳定性不如现有 variable-budget multiclass planner。暂时不作为主结果。

## 当前主线更新

当前最好可写的主方法仍是：

**Runtime variable-budget risk-constrained KV planner + RoPE-aware KV repack**

可作为附加安全 baseline：

**min-safe variable-budget checkpoint**

不建议作为主结果：

- cost-sensitive CE 初版；
- post-hoc temperature；
- action-level safety verifier 初版。

## 下一步

更有希望的下一步不是继续调 CE 权重，而是：

1. 扩大 action-level 数据量：
   - LongBench m=30/m=50；
   - 加 8k/16k RULER；
   - 增加不同 seeds / context placements。
2. 做 calibrated conformal-style fallback：
   - 在校准集上估计每个 action 的 empirical risk；
   - 选择满足风险上界的最小预算；
   - 这比直接学 `p(safe)` 更稳定。
3. 接入 runtime 时保留两个策略：
   - aggressive best-label：低 KV；
   - conservative min-safe：高安全。

论文表述上可以把本轮负结果作为 ablation：

**普通 cost-sensitive CE 和独立 safety verifier 不足以解决 worst-case 风险；有效方案需要把 oracle/worst-case 标签、tail-risk 校准和 runtime KV repack 联合起来。**
