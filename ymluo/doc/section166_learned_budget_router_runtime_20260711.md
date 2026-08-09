# Section 166: Learned Budget Router 在线化与下一步预算候选池

日期：2026-07-11

## 目标

把 v300 里依赖人工 task policy / action router 的部分，推进到“模型按样本预测最小安全预算/动作”。这一步的核心不是继续手写每个任务用多少 budget，而是让 router 从当前样本的长度、block score 分布、任务 family/task 等特征里预测。

## 已完成

1. 新增 `learned_budget_v1` 在线 action router。
   - 代码入口：`src/run_controlled_public_kv_benchmark_v1.py`
   - 新参数：
     - `ours_learned_router_model_path`
     - `ours_learned_router_action_policy_json`
     - `ours_learned_router_confidence_threshold`
     - `ours_learned_router_default_action`
   - 输出新增：
     - `ours_action_router_raw_action`
     - `ours_action_router_selected_action`
     - `ours_action_router_confidence`
     - `ours_action_router_fallback_reason`
     - `ours_action_router_policy_overrides`

2. 新增 task policy 继承机制。
   - `__extends`：继承已有 policy。
   - `__overlay_all_tasks`：对继承 policy 中的所有具体任务统一覆盖字段。
   - 这样 v332/v340 系列不需要复制整份 v300 policy。

3. 第一版在线 smoke 跑通。
   - `v332_learned_budget_router_smoke`：conf=0.35，基本都 low-confidence fallback 到 reference。
   - `v333_learned_budget_router_conf0_smoke`：conf=0，模型开始选择动作。
   - M5 四任务对齐 v300：
     - score：0.346665 vs v300 0.346998，差 -0.000333
     - online：1.129s vs v300 1.217s，约 1.08x
     - KV：48.46% vs v300 47.36%，没有省 KV，反而略高

## 关键发现

### 1. v1 offline replay 不能直接当最终在线结果

v1 使用了 `ours_query_coverage_*`，这些特征来自“已经选完 block 之后”的统计。在线 action router 在选 block 前决策，所以不能严格使用这些 post-selection 特征。当前在线实现用 top-scored pages 估计 coverage，但这会造成分布偏移。

### 2. runtime-clean v2/v3 更可部署，但收益有限

v2 使用 preselection features + family；v3 使用 preselection features + family + task。

代表性 offline replay：

| router | split | score ratio vs v300 | KV keep | speed vs v300 | safe rate |
|---|---:|---:|---:|---:|---:|
| v2 family conf0.65 | all | 99.60% | 26.69% | 1.004x | 99.69% |
| v3 family+task conf0.50 | all | 98.58% | 25.69% | 1.078x | 97.19% |
| v3 family+task conf0.65 | all | 99.18% | 26.78% | 1.001x | 99.56% |

结论：当前已映射 action 候选池主要是 short-decode / speed patch / bridge，不是真正的低 KV 预算候选。因此 router 即使能学，也很难显著降低 KV。

### 3. 不能直接把 v324/v325/v326 等激进候选纳入主 action map

这些候选虽然降低部分任务 KV，但质量损失明显：

| method | overall score | KV keep | 主要问题 |
|---|---:|---:|---|
| v324 | 0.43398 | 25.51% | qasper 从 0.4236 降到 0.3631，narrativeqa 也降 |
| v325 | 0.43596 | 26.67% | qasper 从 0.4236 降到 0.3711 |
| v326 | 0.43805 | 27.24% | narrativeqa 从 0.1960 降到 0.1771 |
| v329 | 0.43508 | 27.17% | 2wikimqa 从 0.4444 降到 0.3780 |

这些方法可以作为训练负例或 oracle 对照，不能作为当前主方法正例。

## 当前正在跑

已在服务器后台提交 v300 operator 不变、只覆盖全任务预算的 LongBench M20 sweep：

| policy | budget |
|---|---:|
| v340 | 256 |
| v341 | 384 |
| v342 | 512 |
| v343 | 768 |
| v344 | 1024 |
| v345 | 1536 |
| v346 | 2048 |
| v347 | 3072 |

这些结果完成后，下一步训练真正的 budget router：

1. 对每个样本枚举 budget candidates。
2. 以 v300 分数为质量下限，选择最小 safe budget。
3. 用 preselection features 训练 `budget_tokens` 分类器。
4. 在线只预测预算，不改变 operator，先验证是否能在不掉分前提下降 KV。
5. 如果预算 router 稳，再把 short-decode / bridge 作为第二阶段 action 加进去。

## 当前判断

目前“模型自己找预算”已经跑通了工程链路，但还不是最终主结果。真正有希望的方向不是在现有 v300/v311/v329/v330 action 里硬选，而是用正在跑的预算 sweep 生成干净的 budget labels，再训练 sample-level budget planner。

## 2026-07-11 追加：budget sweep 已完成后的计划

服务器上 B=256/384/512/768/1024/1536/2048/3072 的 M20 sweep 已完成。对齐同一批 320 个样本和 v300 后：

| method | score | v300 score | score ratio | KV keep | v300 KV | speed vs v300 |
|---|---:|---:|---:|---:|---:|---:|
| B256 | 0.4159 | 0.4461 | 93.23% | 24.97% | 25.95% | 1.099x |
| B384 | 0.4325 | 0.4461 | 96.95% | 26.39% | 25.95% | 1.029x |
| B512 | 0.4372 | 0.4461 | 97.98% | 28.64% | 25.95% | 1.022x |
| B768 | 0.4328 | 0.4461 | 97.02% | 32.43% | 25.95% | 0.856x |
| B1024 | 0.4351 | 0.4461 | 97.51% | 35.81% | 25.95% | 0.988x |
| B1536 | 0.4477 | 0.4461 | 100.35% | 42.73% | 25.95% | 0.885x |
| B2048 | 0.4420 | 0.4461 | 99.07% | 48.36% | 25.95% | 0.790x |
| B3072 | 0.4509 | 0.4461 | 101.06% | 58.38% | 25.95% | 0.694x |

Oracle 最小安全预算上界：

| selector | score | v300 score | KV keep | v300 KV | speed |
|---|---:|---:|---:|---:|---:|
| per-sample budget oracle | 0.4528 | 0.4461 | 24.10% | 25.95% | 1.044x |

这个上界说明 clean budget router 有潜力，但 v300 本身已经很强、平均 KV 已很低，所以预算 router 的增益不会是数量级提升。它的价值更像“把人工 v300 policy 自动化，并在局部样本上进一步找最小安全预算”。

已在本地准备好 v4 文件：

- `scripts/train_clean_budget_router_v4_20260711.py`
- `scripts/summarize_clean_budget_router_v4_20260711.py`
- `configs/riskkv_learned_budget_action_policy_v4_20260711.json`
- `configs/riskkv_task_policy_v348_clean_budget_router_v4_conf050_20260711.json`

当前阻塞：服务器 SSH 端口暂时连接超时。恢复后执行：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/train_clean_budget_router_v4_20260711.py
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_clean_budget_router_v4_20260711.py
```

如果 v4 offline held-out 指标合理，再用 `riskkv_task_policy_v348_clean_budget_router_v4_conf050_20260711.json` 跑在线 M20 验证。
