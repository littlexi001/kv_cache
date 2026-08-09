# section175：v378 sample-level policy-action planner

日期：2026-07-12

## 动机

v377 把 M20 里观察到的任务级 Pareto 现象手工组合起来：少数证据闭包任务允许更高预算，其它任务用极低 KV 抵消全局平均 KV。但任务级组合仍然粗糙，风险在于：

- 同一个任务内部样本难度差异很大，固定任务策略会浪费 KV 或误伤困难样本。
- v360/v363/v365 这些强候选本身继承了 v293 sample-level risk rules，不能只把它们当成静态 budget。
- 如果 v377 在 M100 上不稳，需要一个更通用的 sample-level router，而不是继续手工拼任务。

## 方法

新增 v378：sample-level cost-aware policy-action planner。

核心思想：把一个候选动作定义为“完整 policy family”，而不是单个预算。router 对每个样本预测候选 policy 是否安全，然后按估计 KV 成本从小到大选择第一个安全 policy。

候选 policy 包括当前已完成或即将完成的：

- `policy_v360`: low-KV certificate
- `policy_v363`: taskwise low-KV mix
- `policy_v365`: ultra skeleton
- `policy_v368`: direct operator extreme mix
- `policy_v373`: selective direct ladder
- `policy_v375`: Pareto fused low-KV
- `policy_v376`: strict10 Pareto fused
- `policy_v377`: global Pareto knapsack

训练标签使用 full KV 分数定义安全性：

`safe(action, x) = 1[score(action, x) >= max(0.95 * score_full(x), score_full(x) - 0.05)]`

运行时 fallback 不是 full KV，而是 `policy_v365 + v293_rules`，这样不会把平均 KV 拉爆。选中 learned policy 后继续接 `v293_rules`，用于复现旧候选里已经有效的风险升级逻辑。

## 已新增文件

- `ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/train_policy_action_planner_v378_20260712.py`
- `ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/watch_v378_policy_action_planner_20260712.sh`

训练脚本会自动输出：

- `outputs/riskkv_v19_policy_action_planner_v378_20260712/model.pkl`
- `outputs/riskkv_v19_policy_action_planner_v378_20260712/action_policy.json`
- `outputs/riskkv_v19_policy_action_planner_v378_20260712/planner_summary.csv`
- `configs/riskkv_task_policy_v378_policy_action_planner_20260712.json`

watcher 逻辑：

1. 训练 v378 router。
2. 跑 v378 M20。
3. 若 M20 满足 `score/full >= 95%`、`KV <= 10.5%`、`speed/full >= 2.5x`，自动跑 M100。

## 下一步

服务器 SSH 恢复后：

1. 同步 v378 脚本。
2. 确认 v375/v376/v377 是否已经完成。
3. 启动 `scripts/watch_v378_policy_action_planner_20260712.sh`。
4. 若 v378 M20/M100 成功，则把论文故事从“task-level budget policy”升级为“sample-level policy-action planner with low-KV safety certificate”。
