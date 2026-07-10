# Section 113: Coverage-MMR Memory Action Probe

## 动机

当前最强实际方法是：

```text
evidence-flow page scoring
+ task-family minimum-safe action
+ benefit-calibrated conformal risk gate
+ selective counterfactual consistency verifier
```

这条线主要解决“什么时候 sparse action 不安全”。下一步需要继续增强“sparse action 本身选哪些 KV pages”。

原来的 MMR 主要做两件事：

```text
选择高相关 page
惩罚语义冗余 page
```

但它没有显式优化“query 中不同证据锚点是否都被覆盖”。这会导致一个常见失败模式：

```text
多个 selected pages 都围绕同一个高分实体或局部段落，
但问题需要另一个人名、数字、标题或第二跳实体。
```

## Coverage-MMR

新增一个非 oracle、推理时可用的组合选择项。

先从 query 中提取 coverage terms：

```text
C(q) = title anchors ∪ named entities ∪ numbers ∪ non-stopword content words
```

每个候选 KV page 有：

```text
C_i = C(q) ∩ terms(block_i)
```

在第 `t` 轮 page selection 时，已有选择集合为 `S_t`，已覆盖锚点为：

```text
U_t = union_{j in S_t} C_j
```

候选 page 的 MMR 分数变为：

```text
score_i =
  lambda * relevance_i
  - (1 - lambda) * redundancy_i
  + beta * |C_i \ U_t| / max(1, |C(q)|)
```

其中：

```text
beta = ours_coverage_mmr_weight
```

默认 `beta=0`，所以不会影响旧实验。v65 probe 使用：

```text
coverage_mmr_weight = 0.20
coverage_mmr_max_terms = 32
```

## 为什么这不是普通 RAG

这个机制不是简单 top-k retrieval。它优化的是 KV-cache memory action：

```text
给定固定 KV token budget，
选择一组可直接组成 compact KV cache 的 pages，
并让这组 pages 同时满足 relevance、non-redundancy、anchor coverage 和 risk control。
```

普通 RAG 更像“返回若干文档片段”；这里的选择结果会进入 RoPE-aware / page-gather KV decode 路径，且和 fallback、verifier、conformal gate 共同组成 memory-action policy。

## 当前实验

新增代码：

```text
src/run_controlled_public_kv_benchmark_v1.py
```

新增 per-sample 诊断字段：

```text
ours_query_coverage_terms
ours_query_coverage_covered
ours_query_coverage_recall
```

这些字段用于后续训练或校准 coverage-risk gate：

```text
if selected action covers too few query anchors:
    expand sparse budget or fallback to a safer memory action
```

新增配置：

```text
configs/riskkv_task_policy_v65_coverage_mmr_benefit_conformal_qasper_full_20260709.json
```

新增运行脚本：

```text
scripts/run_riskkv_v65_coverage_mmr_m20_20260709.sh
```

服务器启动：

```text
output: outputs/riskkv_v65_coverage_mmr_benefit_conformal_qasper_full_m20_20260709
log:    logs/riskkv_v65_coverage_mmr_benefit_conformal_qasper_full_m20_20260709.log
gpu:    7
```

v65 对照对象：

```text
v64 = benefit-conformal + qasper full fallback
v65 = v64 + Coverage-MMR
```

## v66: Task-Scoped Coverage-MMR

v65 是全局打开 coverage bonus，可能会影响 summarization/code 这类不一定依赖 query anchor 覆盖的任务。因此增加一个更稳的 v66：

```text
v66 = v64 + task-scoped Coverage-MMR
```

只在下面任务打开：

```text
narrativeqa
multifieldqa_en
2wikimqa
passage_retrieval_en
```

其中前三个是 QA 风险控制任务，`passage_retrieval_en` 是结构化检索任务。summarization、code、classification 和已经 full fallback 的任务不打开 coverage bonus。

新增配置：

```text
configs/riskkv_task_policy_v66_task_coverage_mmr_benefit_conformal_qasper_full_20260709.json
```

新增运行脚本：

```text
scripts/run_riskkv_v66_task_coverage_mmr_m20_20260709.sh
scripts/watch_and_launch_v66_after_v65_gpu7_20260709.sh
```

服务器 watcher 策略：

```text
先等 v65 summary
再等 GPU7 空闲
然后自动启动 v66
```

如果 v65 提升，说明当前 bottleneck 不是只有 router/risk gate，也包括 sparse memory action 的组合覆盖质量；这可以作为论文中更强的“action optimization”创新点。

如果 v65 下降，则说明 query-level anchor coverage 太粗，下一步应该改成：

```text
coverage bonus 只在 multi-hop QA / retrieval tasks 打开
或用 task-family beta_g 校准 coverage strength
```

v66 正是对这个风险的提前验证。

## v65 result: global Coverage-MMR

v65 已完成 m20：

| Setting | Score | KV ratio | Online |
|---|---:|---:|---:|
| v53 consistency+qasper | 0.375890 | 62.89% | 2.736s |
| v65 global Coverage-MMR | 0.375736 | 60.89% | 2.792s |

结论：

```text
Global Coverage-MMR is not a clear quality win.
It slightly reduces KV ratio but does not improve score and is slower in the current harness.
```

这不是完全失败：它说明 coverage signal 没有破坏整体质量，但全局打开不够精准。
更合理的方向是：

```text
1. task-scoped coverage bonus, tested by v66;
2. coverage-risk gate, tested by v67/v68;
3. calibrated coverage threshold tau_g, prepared by calibrate_coverage_risk_gate_20260709.py.
```

## v67/v68: Pre-decode Coverage-Risk Gate

Coverage-MMR 改的是“如何选 pages”。进一步的创新是把 coverage 从一个 reranking bonus 变成 memory-action safety certificate：

```text
先按当前 sparse policy 选 KV pages
计算 selected pages 对 query coverage terms 的覆盖率
如果覆盖率过低，在 decode 前升级到更安全的 memory action
```

定义：

```text
rho(x, a) = | union_{i in selected(a)} C_i | / max(1, |C(q)|)
```

当满足：

```text
|C(q)| >= M_g
rho(x, a) < tau_g
```

触发：

```text
a <- expanded sparse action, e.g. 2048-token KV budget
```

当前 probe 设置：

```text
QA tasks:
  tau_g = 0.25
  M_g   = 4
  safe action = 2048-token sparse KV

passage_retrieval_en:
  tau_g = 0.20
  M_g   = 3
  safe action = 2048-token sparse KV
```

新增配置：

```text
configs/riskkv_task_policy_v67_coverage_risk_gate_qasper_full_20260709.json
configs/riskkv_task_policy_v68_coverage_mmr_risk_gate_qasper_full_20260709.json
```

新增运行脚本：

```text
scripts/run_riskkv_v67_v68_coverage_risk_m20_20260709.sh
scripts/watch_and_launch_v67_v68_after_v66_20260709.sh
```

新增校准脚本：

```text
scripts/calibrate_coverage_risk_gate_20260709.py
```

用途：

```text
给定 base policy 和 reference/safe policy 的成对 task_results.csv，
按 task family 学习 coverage recall 阈值 tau_g，
并输出 stitched score / KV / online 估计。
```

后续如果 v67/v68 有正信号，可以用该脚本生成 calibrated v69，而不是继续手调阈值。

## v69: Calibrated Coverage-Risk Policy

为了避免 v67/v68 停留在手调阈值，新增自动校准和启动流程：

```text
scripts/make_coverage_calibrated_policy_20260709.py
scripts/run_riskkv_v69_calibrated_coverage_m20_20260709.sh
scripts/watch_calibrate_and_launch_v69_coverage_20260709.sh
```

Watcher 等待：

```text
base:      v66 task-scoped Coverage-MMR
reference: v64 benefit-conformal qasper-full
```

两者完成后自动运行：

```text
calibrate_coverage_risk_gate_20260709.py
make_coverage_calibrated_policy_20260709.py
```

生成：

```text
configs/riskkv_task_policy_v69_calibrated_coverage_mmr_qasper_full_20260709.json
outputs/coverage_risk_calibration_v66_vs_v64_m20_20260709.csv
```

然后等空闲 GPU 启动：

```text
outputs/riskkv_v69_calibrated_coverage_mmr_qasper_full_m20_20260709
```

校准策略不是无条件打开 coverage-risk：

```text
只保留 beneficial >= 1 且 trigger_rate <= 75% 的 task 阈值
```

这样可以防止 coverage-risk 退化为“很多样本都加预算”的粗暴策略。

对照关系：

```text
v67 = v64 + coverage-risk gate
v68 = v66 + coverage-risk gate
```

要看的核心问题：

```text
1. v67 > v64: coverage certificate 本身能识别 unsafe sparse action
2. v68 > v66: Coverage-MMR 和 coverage-risk gate 可以叠加
3. v68 token ratio 不显著高于 v64/v66: safety certificate 不只是粗暴加预算
```
