# Section 79：Worst-case targeted planner 评估

日期：2026-07-06

## 目的

验证 `ranker_or_task_rule_gap_fallback` 是否能直接作为最终方法使用，尤其是在更容易失败的 exact / retrieval 场景上。

这次不是全量 benchmark，而是 targeted worst-case smoke：

- LongBench hard exact/retrieval：`hotpotqa`、`2wikimqa`、`musique`、`passage_retrieval_en`、`passage_count`
- RULER 4k/8k hard：`niah_multiquery`、`niah_multivalue`、`vt`、`cwe`、`fwe`
- RULER 16k hard：`niah_single_2`、`niah_multikey_1`、`niah_multiquery`、`niah_multivalue`、`vt`、`cwe`、`fwe`

每个任务跑 8 个样本，候选动作 16 个。

## 输出路径

- benchmark 输出：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706`
- planner 评估：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/planner_eval_v1`

## Planner 结果

测试 split：63 个样本。训练 split：113 个样本。

| 策略 | 相对 full 质量 | token 比例 | success |
| --- | ---: | ---: | ---: |
| `runtime_ranker` | 0.8113 | 0.1368 | 0.7937 |
| `length_aware_rule` | 1.0000 | 0.3647 | 0.9206 |
| `ranker_or_rule_gap_fallback` | 1.0189 | 0.3686 | 0.9365 |
| `ranker_or_task_rule_gap_fallback` | 0.9811 | 0.2027 | 0.9048 |
| `oracle_budget_oracle_action` | 1.0566 | 0.1353 | 1.0000 |

## 结论

`ranker_or_task_rule_gap_fallback` 不能直接作为最终方法使用。它不是 oracle，也不是稳定的 fully learned router。它是一个可执行的 hybrid 原型，但 task fallback rule 在 worst-case 上有明显过拟合。

最重要的发现：

1. action space 仍然很强。oracle 在 worst-case 上仍有 `1.0566x` full 质量、`13.5%` token、`100%` success。
2. 当前 runtime ranker 明显不稳，只有 `0.8113x` full、`79.4%` success。
3. 原来的 task-rule fallback 过于激进，把一些 exact task 错配到便宜但错误的动作上。
4. 保守的 `ranker_or_rule_gap_fallback` 更稳，达到 `1.0189x` full，但 token 比例高到 `36.9%`。

## 主要失败模式

### `niah_multivalue`

当前 task rule 有时选择 `static_hier`，但 worst-case 上会失败。更稳的候选是：

- `retrieval_raw_k1`
- `recent_plus_retrieval_raw_k1`
- 必要时 `retrieval_raw_k2`

### 16k `niah_single_2`

`recent_plus_retrieval_raw_k2` 在部分样本上失败。`k3` 能修复：

- `recent_plus_retrieval_raw_k3` 成功，token 比例约 0.261
- `retrieval_raw_k3` 成功，token 比例约 0.292

### LongBench `musique`

某些 multi-hop 样本中，retrieval k1 会失败，`summary1_4` 或 `recent_plus_summary1_4` 反而成功。

## 当前判断

现在最好的“研究方向”仍然是：

> typed summary KV + retrieval/recent fallback + two-stage risk-aware planner

但当前最好的“可报告算法”还需要继续做成：

1. 更稳的 risk classifier，判断什么时候不能用 summary/static/recent-only。
2. task-aware fallback 不能手写死规则，要蒸馏成可泛化 policy。
3. exact/retrieval 任务里要允许动态 k，尤其是 k1/k2/k3，而不是固定 k2。
4. LongBench multi-hop 要区分 retrieval 型和 reasoning/summary 型，因为有些 case summary1_4 比 retrieval 更稳。

下一步应该训练一个 `risk-aware router v2`：

- 输入：当前 router features + task family + retriever score gap + top-k stability features。
- 输出：
  - risk score：当前动作是否可能低于 full；
  - minimal safe action：`summary1_8` / `summary1_4` / `retrieval_k1` / `retrieval_k2` / `retrieval_k3` / full fallback。
- 训练标签：用这次 targeted benchmark + 之前 full combined benchmark 的 oracle/success labels。

目标不是追求最低 token，而是先达到：

- worst-case success >= 98%
- quality >= full
- token ratio <= 25%-30%

这个目标比现在更适合写进 ICLR 主实验。
