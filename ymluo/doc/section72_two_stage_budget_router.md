# Section 72: Two-stage Budget Router 快速实验

## 目标

本节测试一个新的 router 结构：

```text
stage 1: 先预测当前 query/task 需要多少 active KV budget
stage 2: 再在这个 budget 内选择具体 memory action
```

动机是：直接让 router 在 `summary1_8 / summary1_4 / retrieval_raw_k1 / retrieval_raw_k2 / ...` 之间分类太难，尤其长文本下应该先判断任务难度和预算，再决定用多强的策略。

## 代码

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_two_stage_budget_router.py
```

输入是已有 Qwen3-8B benchmark trials，不重新跑模型生成：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_adapter_router_20260705
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_extra_k348_20260705
```

候选方法：

```text
full_raw
recent_only
static_hier
summary1_8
summary1_4
summary1_2
retrieval_raw_k1
retrieval_raw_k2
retrieval_raw_k3
retrieval_raw_k4
retrieval_raw_k8
```

budget bins：

```text
20%, 30%, 40%, 50%, 100%
```

## Oracle label 构造

对于每个 case：

1. 先看 full_raw score。
2. 找到最小 budget bin，使得该 budget 内至少有一个方法达到 full_raw performance。
3. 在这个 budget 内选 token ratio 最低、速度更快的方法作为 oracle action。

这相当于学习：

```text
这个 query 至少需要多少 active KV 才能接近 full_raw？
在这个 budget 内应该用哪个 action？
```

## 数据规模

```text
examples = 128
train = 82
test = 46
```

oracle budget 分布：

| budget | count |
|---:|---:|
| 0.2 | 91 |
| 0.3 | 18 |
| 0.4 | 6 |
| 0.5 | 5 |
| 1.0 | 8 |

oracle action 分布：

| action | count |
|---|---:|
| recent_only | 48 |
| summary1_8 | 29 |
| retrieval_raw_k1 | 21 |
| retrieval_raw_k2 | 14 |
| static_hier | 9 |
| summary1_4 | 2 |
| retrieval_raw_k3 | 2 |
| full_raw | 1 |
| retrieval_raw_k4 | 1 |
| retrieval_raw_k8 | 1 |

这个分布说明：oracle 不是简单“长文本用更大 k”。很多样例的最优动作是 `recent_only` 或很小 budget，因此第二阶段 action 选择比预算预测更难。

## 第一版：budget head + action head

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_budget_router_rule_20260705
```

test overall：

| policy | budget acc | action acc | relative to full | token ratio |
|---|---:|---:|---:|---:|
| two_stage_masked | 78.26% | 60.87% | 81.68% | 17.18% |
| pred_budget_oracle_action | 78.26% | 89.13% | 94.71% | 15.23% |
| oracle_budget_oracle_action | 100.00% | 100.00% | 105.14% | 18.01% |

解释：

- `two_stage_masked` 是真正 runtime 可用版本：预测 budget，再用 action head 在 budget 内选 action。
- `pred_budget_oracle_action` 是分析上界：budget 用模型预测，但 stage 2 用 oracle 选 budget 内最优 action。
- `oracle_budget_oracle_action` 是两阶段 oracle 上界。

关键结论：

```text
budget stage 是有价值的：
  只要 stage 2 能在预测 budget 内选好 action，就能达到 94.7% full_raw、15.2% active KV。

当前瓶颈是 stage 2 action selection：
  action head 只有 60.9% test action accuracy，导致实际 two_stage_masked 掉到 81.7% full_raw。
```

## 第二版：budget + rule action

策略：

```text
generation: 预算内优先选 summary1_8 / summary1_4 / summary1_2
exact: 预算内选最大 retrieval k
```

test overall：

| policy | relative to full | token ratio |
|---|---:|---:|
| pred_budget_rule_action | 66.16% | 20.78% |
| oracle_budget_rule_action | 73.94% | 21.46% |

结论：

- 简单规则不可靠。
- “预算内选最大 k”会引入噪声，不一定提升 exact task。
- 这说明 stage 2 不能只看 budget，还必须看 query 和证据分布。

## 第三版：budget + score ranker

策略：

```text
stage 1: 预测 budget
stage 2: 对 budget 内每个 candidate action 预测 utility/score，再选最高分
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_budget_router_ranker_20260705
```

test overall：

| policy | budget acc | action acc | relative to full | token ratio |
|---|---:|---:|---:|---:|
| pred_budget_score_ranker | 80.43% | 36.96% | 77.06% | 17.82% |
| oracle_budget_score_ranker | 100.00% | 43.48% | 82.88% | 20.10% |
| pred_budget_oracle_action | 80.43% | 89.13% | 94.19% | 14.52% |
| oracle_budget_oracle_action | 100.00% | 100.00% | 105.55% | 18.40% |

结论：

- score ranker 在 train 上能拟合，但 test 上泛化差。
- 当前 128 条 benchmark trial 太少，无法训练出可靠的 action utility model。
- 但 `pred_budget_oracle_action` 仍然稳定在约 94% full / 15% active KV，说明预算预测方向是对的。

## 多 seed 结果

使用 `hidden_dim=32, epochs=500` 扫了 4 个 seed，只看 runtime 可用的 `two_stage_masked`：

| seed | relative to full | token ratio |
|---:|---:|---:|
| 2026070511 | 81.78% | 16.39% |
| 2026070512 | 78.48% | 15.45% |
| 2026070513 | 75.14% | 15.23% |
| 2026070514 | 85.75% | 14.50% |

最好的 seed 仍然只有：

```text
85.75% full_raw, 14.50% active KV
```

这低于之前的 `length_aware_rule`：

```text
102.87% full_raw, 40.94% active KV
```

## 当前判断

two-stage budget router 的方向是对的，但当前 quick experiment 还不能替代已有 router。

最重要的发现是：

```text
budget prediction 已经比较有信号；
stage-2 action selection 是主要瓶颈。
```

如果 stage 2 达到 oracle，当前 held-out test 可以做到：

```text
94% - 95% full_raw
14% - 15% active KV
```

这正好接近论文目标。但现在真实 runtime 版本只有：

```text
82% - 86% full_raw
14% - 17% active KV
```

所以不能直接用这版作为最终 router。

## 下一步

建议下一步不要继续只在 128 条 benchmark trial 上调网络结构，而是构造非 benchmark synthetic exact/retrieval 数据训练 stage 2：

1. 生成多 block、多答案、多位置、多干扰证据样例。
2. 对每个样例生成 candidate action labels：

   ```text
   recent_only
   retrieval_raw_k1/k2/k3/k4/k8
   summary1_8/1_4/1_2
   full_raw
   ```

3. 训练：

   ```text
   budget head: 预测所需 active KV budget
   action/ranker head: 在预算内预测哪类 action 能答对
   ```

4. 再只在 LongBench/RULER 上测试，不用 benchmark 数据蒸馏。

简短结论：

```text
two-stage budget router 值得保留；
但这次 quick test 证明，不能只改 router 结构，还需要专门训练 stage-2 action selector。
```
