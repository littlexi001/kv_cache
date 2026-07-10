# Section 107: RiskKV-Block v1.9 动态动作策略与安全回退

日期：2026-07-09

## 当前判断

v1.8 的 `qasper,musique` bridge gate 在 fast m6 子集上很好，但 m20 全任务验证暴露出一个问题：固定 `budget=512,page=128` 不适合所有 LongBench 任务。尤其是 synthetic retrieval、passage counting、summarization 和部分科学 QA，对预算或安全回退更敏感。

因此下一步不应该继续只调 bridge fraction，而应该把方法从“固定预算 block routing”推进到：

```text
query/task/risk features -> KV action policy -> compact decode action
```

其中 action 不只是选择 scorer，还包括：

- 使用多少 KV budget；
- 使用多大的 block/page size；
- 是否启用 bridge expansion；
- 是否对高风险任务执行 full-cache fallback。

这更符合此前 risk-aware router 的故事，也能避免 Table5 固定 B 设置把方法限制住。

## 已修复的问题

### 1. Bridge gate learned trainer

脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/train_bridge_gate_from_labels_20260709.py
```

原来用 `ColumnTransformer` 直接吃 `list[dict]`，sklearn 会报一维输入错误。已经改为：

```text
DictVectorizer -> StandardScaler -> LogisticRegression
```

服务器 fast m6 重新训练结果：

```text
examples: 36
no_bridge_score:      0.366116
all_bridge_score:     0.370280
task_policy_score:    0.398057
sample_oracle_score:  0.398057
learned_train_accuracy: 0.750000
learned_train_policy_score: 0.398057
learned_loo_accuracy: 0.722222
learned_loo_policy_score: 0.384168
```

当前 task-level gate 仍是：

```text
bridge: qasper, musique
no bridge: 2wikimqa, hotpotqa, lcc, passage_retrieval_en
```

### 2. Runner 支持 per-task action policy

脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_controlled_public_kv_benchmark_v1.py
```

新增参数：

```text
--ours_task_policy_json
--ours_full_fallback_tasks
```

`--ours_task_policy_json` 可以传 JSON 文件或 JSON 字符串。每个 task 可以覆盖：

```text
budget_tokens
sink_tokens
recent_tokens
page_tokens
ours_scorer
ours_bridge_tasks
bridge
full_fallback
ours_* scalar routing params
```

默认行为完全不变；只有传 policy 时才会按样本生成 effective config。

### 3. Full fallback 不再额外 gather

如果某个 action 保留全部 prefix KV，现在会直接复用 full prefix cache：

```text
if len(keep_indices) >= prefix_length:
    sparse_cache = full_prefix_cache
    gather_seconds = 0
```

这样 full fallback 的测速不会被一次无意义的 full gather 污染。

## v1.9 新策略

### Safe policy

文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v19_safe_20260709.json
```

核心规则：

```text
default: budget=512, page=128, multiscale-flow
qasper: budget=1024 + task bridge
musique: budget=512 + task bridge
passage_count: full fallback
passage_retrieval_en: full fallback
```

目的：验证“最小安全动作”能否显著修复 synthetic / counting 类任务，同时保留大部分任务的压缩收益。

### Budget-only policy

文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v19_budget_20260709.json
```

核心规则：

```text
default: budget=512, page=128, multiscale-flow
qasper: budget=1024 + task bridge
musique: budget=512 + task bridge
gov_report/qmsum/multi_news: budget=1024
passage_count/passage_retrieval_en: budget=1024
```

目的：区分提升是否必须依赖 full fallback，还是动态预算本身就能解决一部分问题。

## 当前已知结果

### fast m6 bridge fraction sweep

```text
bridge_fraction=0.12: score 0.395663
bridge_fraction=0.14: score 0.384169
bridge_fraction=0.18: score 0.398057
bridge_fraction=0.20: score 0.398057
baseline v18 0.16:   score 0.398057
```

结论：`qasper,musique` bridge gate 的收益不是单个 fraction 偶然点，`0.16-0.20` 区间基本稳定；`0.14` 偏低。

### m20 固定 512 结果

```text
v13 multiscale-flow m20:        0.262113
v18 broad task-bridge m20:      0.265749
v18 qasper/musique-only m20:    0.267311
v16 all-bridge m20:             0.258415
```

结论：固定 512 的 m20 全任务分数不够好，不能作为最终主方法。这不是 bridge 一项的问题，而是固定预算/固定动作与任务风险不匹配。bridge 本身也必须路由：all-bridge 是负的，qasper/musique-only 是正的。

### v1.9 safe policy m6

输出：

```text
outputs/riskkv_v19_safe_20260709_task_policy_v19_fixfull_m6_bDyn_pDyn
```

结果：

```text
score:       0.302213
token ratio: 23.62%
online:      2.544s
```

关键任务：

```text
passage_count:        0.166667, keep=100%, kv_gather=0
passage_retrieval_en: 0.333333, keep=100%, kv_gather=0
qasper:               0.518569, budget=1024, keep=28.87%
```

结论：full fallback 作为 minimum-safe action 是有效的，尤其 synthetic/counting 类任务明显修复；但 safe policy 没给 summarization 增加预算，整体分数仍受 `gov_report/qmsum/multi_news` 拖累。因此需要继续比较 budget-only 和 safe-budget combo。

### v1.9 budget-only / safe-budget m6

budget-only：

```text
score:       0.298580
token ratio: 15.31%
online:      2.664s
```

safe-budget：

```text
score:       0.305525
token ratio: 26.05%
online:      2.596s
```

safe-budget 是当前单个 hand-written policy 里最高的，但比 safe 只高 `+0.0033`，token ratio 多 `+2.43%`，说明“给 summarization 加 1024 budget”的收益很小。

### v1.9 多动作蒸馏

输入动作：

```text
budget
safe
safe_budget
```

蒸馏结果：

```text
sample_oracle_score: 0.319654
task_policy_score:   0.308997
```

这说明还有可学习空间。task-level policy 的主要结论是：以 budget-only 为底座，但 `passage_retrieval_en` 更适合 full fallback；`passage_count` 不应该 full fallback，只需要 1024 budget。

### v2.0 budget-retrieval-safe policy

文件：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/configs/riskkv_task_policy_v20_budget_retrieval_safe_20260709.json
```

规则：

```text
default: budget=512, page=128, multiscale-flow
qasper: budget=1024 + bridge
musique: bridge
gov_report/qmsum/multi_news: budget=1024
passage_count: budget=1024
passage_retrieval_en: full fallback
```

这是由 v19 三个策略的多动作蒸馏得到的更克制版本。m6 验证已排队：

```text
outputs/logs/riskkv_task_policy_v20_budget_retrieval_safe_20260709.master.log
```

验证结果：

```text
score:       0.308997
token ratio: 20.69%
online:      2.625s
```

关键任务：

```text
gov_report:            0.173941, budget=1024
multi_news:            0.146033, budget=1024
qmsum:                 0.133962, budget=1024
qasper:                0.518569, budget=1024 + bridge
passage_count:         0.222222, budget=1024
passage_retrieval_en:  0.333333, full fallback
```

当前结论：v2.0 是 fast m6 上最好的实际策略，比 `safe_budget` 更高且 token 更少：

```text
budget-only:  0.298580, token 15.31%
safe:         0.302213, token 23.62%
safe-budget:  0.305525, token 26.05%
v2.0 policy:  0.308997, token 20.69%
```

这说明“minimum-safe action”不应该简单地对所有 synthetic/counting 任务 full fallback，而应该更细：`passage_retrieval_en` full fallback，`passage_count` 用 1024 sparse budget 即可。

## m20 动态 action policy 初步结果

完整 LongBench 16 任务、每任务 20 样本：

```text
fixed 512 qasper/musique bridge: 0.267311, token 11.78%
safe policy:                    0.316013, token 24.06%
safe-budget policy:             0.318159, token 26.32%
v2.0 cost-strict policy:        0.312950, token 20.99%
```

这说明动态 action routing 在 m20 上不是 fast m6 偶然点，已经从固定 512 的 `0.2673` 提升到 `0.3182`。

当前两个最终候选：

```text
v22-quality:     0.318159, token 26.32%
v22-cost-strict: 0.312950, token 20.99%
```

`v22-quality` 是当前分数最优策略；`v22-cost-strict` 少用约 5.33% token，但低 0.0052 分。对于论文主结果，优先使用 `v22-quality`；对于 Pareto/frontier 图，保留 `v22-cost-strict` 作为高效率点。

## v2.1 retrieval fallback sweep

为了验证 `passage_retrieval_en` 是否必须 full fallback，我跑了 targeted m10 子集：

```text
tasks = passage_count, passage_retrieval_en, qasper, musique
samples_per_task = 10
```

对 `passage_retrieval_en` 尝试：

```text
budget=1024: score 0.300, token 13.96%
budget=1536: score 0.300, token 20.77%
budget=2048: score 0.300, token 27.57%
full fallback: score 0.600, token 100.00%
```

结论：`passage_retrieval_en` 的瓶颈不是 sparse budget 不够，而是 sparse scorer/selected evidence 对这个任务仍然不可靠。full fallback 是当前必要的 minimum-safe action。相比之下，`passage_count` 用 1024 sparse budget 即可，不需要 full fallback。

注意：m20 上 `passage_count` 和 m10 targeted 子集不完全一致。m20 中 full fallback 能把 `passage_count` 从 v20 的 `0.0667` 提到 `0.1500`，因此质量优先的 v22-quality 仍保留 `passage_count` full fallback；cost-strict 版本才使用 1024 sparse budget。

## 正在跑的实验

v1.9 动态 policy m6：

```text
scripts/run_riskkv_task_policy_v19_20260709.sh
outputs/logs/riskkv_task_policy_v19_fixfull_20260709.master.log
```

启动信息：

```text
STAMP=20260709_task_policy_v19_fixfull
SAMPLES=6
TASKS=all 16 LongBench tasks
policy=safe then budget
```

当前服务器 PID：

```text
1956405
```

额外启动的 combo policy：

```text
outputs/riskkv_v19_safe_budget_20260709_task_policy_v19_safe_budget_fixfull_m6_bDyn_pDyn
outputs/logs/riskkv_task_policy_v19_safe_budget_fixfull_20260709.master.log
```

规则：

```text
summarization: budget=1024
qasper: budget=1024 + bridge
musique: bridge
passage_count / passage_retrieval_en: full fallback
```

## 新增多动作蒸馏工具

脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/distill_task_action_policy_from_results_20260709.py
```

用途：从多个 `task_results.csv` 中蒸馏 action labels，不再局限于 bridge/no-bridge 二分类。输入形如：

```text
--result no_bridge=.../task_results.csv
--result qm_bridge=.../task_results.csv
--result safe_policy=.../task_results.csv
```

输出：

```text
action_labels.csv
task_action_policy.csv
action_policy_summary.json
action_policy_report.md
```

fast m6 烟测：

```text
actions: no_bridge, qm_bridge, all_bridge
sample_oracle_score: 0.398057
task_policy_score:   0.398057
```

同分时按 `--result` 输入顺序选择动作，因此可以把“更小、更简单、更安全”的动作放在前面，实现 minimum-safe tie-break。当前 tie-break 后的策略仍然是：

```text
2wikimqa/hotpotqa/lcc/passage_retrieval_en: no_bridge
musique/qasper: qm_bridge
```

## m20 bridge gate 结论

使用完整 16 个 LongBench 任务，每个任务 20 个样本，比较：

```text
no_bridge:  v13 multiscale-flow
all_bridge: v16 multiscale-bridge
qm_bridge:  v18 qasper/musique bridge gate
```

结果：

```text
no_bridge_score:      0.262113
all_bridge_score:     0.258415
qm_bridge_score:      0.267311
sample_oracle_score:  0.270800
```

逐任务蒸馏结果：

```text
bridge: qasper, musique
no bridge: 其他全部任务
```

关键解释：

- all-bridge 是负的，说明 bridge expansion 不是通用增强项；
- qasper/musique bridge gate 在 m20 上仍然有效；
- sample oracle 只比 task policy 高 `0.0035`，说明 bridge 的主要可解释维度是 task/risk family，而不是当前弱特征能稳定捕捉的逐样本信号。

learned logistic gate：

```text
train_policy_score: 0.264283
loo_policy_score:   0.261615
task_policy_score:  0.267311
```

结论：当前不应把主方法写成黑盒 learned bridge router；更稳的论文叙事是“可解释的 risk-action planner”，learned router 作为辅助/未来扩展。

## Action router 训练结论

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/train_action_router_from_labels_20260709.py
```

输入是 `action_labels.csv`，只使用部署前可得的安全特征：

```text
task, metric, raw_prefix_tokens, page_count,
ours_score_max/mean/gap2/gap3/entropy/positive_fraction
```

在 v19/v20 m6 多动作标签上：

```text
sample_oracle_score: 0.319654
task_policy_score:   0.308997
learned_train_score: 0.309038
learned_loo_score:   0.298049
```

结论同样清楚：当前 learned sample router 有过拟合风险，不能作为论文主结果。主线应使用蒸馏得到的可解释 task-family action policy；sample-level learned router 可以作为“当前特征不够强”的负消融，用来说明方法不是靠任意训练一个分类器调参。

## 下一步决策

1. 等 v2.0 policy 的 m20 结果落盘，判断 m6 结论是否能转移。
2. 用 v2.0 m20、safe-budget m20、safe m20 再做一次多动作蒸馏，确认最终主策略。
3. 如果 v2.0 m20 仍最好，把论文主线锁定为 interpretable risk-action planner。
4. 如果 m20 上 full fallback 太贵或收益不足，继续把 retrieval/counting 拆成更细的风险动作。
5. learned sample router 当前只作为负消融；除非 m20/更大样本 LOO 能超过 task-family policy，否则不作为主方法。

## 论文故事调整

建议把方法命名从单纯的 block scorer 提升为：

```text
RiskKV-Block: Risk-Aware Action Routing over Materialized KV Pages
```

核心贡献可以写成三层：

1. Evidence-flow page scoring：直接证据、局部 support、bridge evidence chain。
2. Dynamic KV action policy：按任务/风险选择 budget、page size、bridge。
3. Minimum-safe fallback：当压缩风险超过阈值时，保留 full KV，避免为了速度牺牲不可恢复的质量。

这样和 RAG 的边界也更清晰：我们不检索外部文档，不重写 prompt，而是在 full-context prefill 后，对已经 materialized 的 KV pages 做动作路由。
