# section177：v380 policy multiclass router

日期：2026-07-12

## 为什么做 v380

v378 的 sample-level safety classifier 已经跑通，但离线结果只从 v365 的 `0.3776 / 8.24% KV` 提升到 `0.3809 / 7.37% KV`，没有超过 v376。

错误分析显示：v378 过度选择 `reference` 和 `policy_v373`，而 oracle 经常选择 `policy_v368` 和 `policy_v376`。这说明问题不是候选 policy 不够，而是 safety classifier 的选择方式不够直接。

## 方法

v380 改成直接蒸馏 oracle action：

1. 对每个样本，在候选 policy 中找 oracle action。
2. 标签不是 `safe/unsafe`，而是 `policy_v360/v363/v365/v368/v373/v376/...` 这样的 multiclass action。
3. 训练 multiclass RandomForest。
4. runtime 用已有 `learned_budget_overlay_v1` 预测 action。
5. 新增 `v293_rules_after_learned` 模式：先应用 learned action 的 policy fragment，再接 v293 risk rules；低置信度 fallback 也接 v293。

这个改动的意义：候选 policy 本身很多依赖 v293 动态风险规则，如果 learned router 选中 policy 后不再运行 v293，就无法复现候选结果。

## 离线结果

| method | score | KV keep | speed/full | oracle match |
|---|---:|---:|---:|---:|
| base v365 | 0.3776 | 8.24% | 5.79x | - |
| v378 safety classifier | 0.3809 | 7.37% | 6.14x | - |
| v380 multiclass router | 0.4163 | 7.92% | 6.25x | 77.81% |
| oracle policy selection | 0.4320 | 7.29% | 6.84x | 100% |

v380 离线结果是目前最接近 oracle 的 sample-level router；如果 runtime M20 能兑现，它会比 v376 更有潜力作为主方法故事。

## 已新增文件

- `src/run_controlled_public_kv_benchmark_v1.py`
  - 增加 `v293_rules_after_learned` overlay 模式。
- `scripts/train_policy_multiclass_router_v380_20260712.py`
- `scripts/watch_v380_policy_multiclass_router_20260712.sh`

训练输出：

- `outputs/riskkv_v19_policy_multiclass_router_v380_20260712/model.pkl`
- `outputs/riskkv_v19_policy_multiclass_router_v380_20260712/action_policy.json`
- `configs/riskkv_task_policy_v380_policy_multiclass_router_20260712.json`

## 当前状态

v380 M20 已提交运行。若 M20 满足：

- `score/full >= 95%`
- `KV <= 10.5%`
- `speed/full >= 2.5x`

watcher 会自动启动 v380 M100。
