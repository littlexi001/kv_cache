# 子问题 3：局部规则在长上下文推理中的失败边界

日期：2026-07-09

## 目标

这个实验直接测试：

```text
如果单步规则只是 if A then B，小模型为什么放到长 context 后仍然失败？
```

核心诊断不是证明模型不会做简单逻辑，而是画出失败边界：

1. 干净短上下文中是否能完成同样规则；
2. context 变长后先在哪些条件下失败；
3. 失败更像检索/筛选退化、变量绑定错误、竞争链混淆，还是中间状态维护失败；
4. 失败时 relevant rule attention selectivity 是否下降。

## 工作负载

每个样本包含一条或多条 verified rule：

```text
VERIFIED RULE T0: IF A IS ACTIVE THEN B BECOMES ACTIVE.
VERIFIED RULE T1: IF B IS ACTIVE THEN C BECOMES ACTIVE.
```

问题要求从 start code 出发，应用固定步数，回答最终 code。

只要 chain length = 1，单步依赖就是最简单的短程关系。chain length > 1 时，复杂性只来自很多局部 step 连续组合。

## 控制变量

| 变量 | 含义 |
|---|---|
| `target_context_tokens` | haystack 主体 token 数 |
| `distractor_count` | 干扰事件数量 |
| `distractor_similarity` | `low/medium/high/conflict` |
| `rule_gap_tokens` | 相邻 relevant rule 的目标间隔 |
| `depth_percent` | relevant rule block 在上下文中的位置 |
| `chain_length` | 局部规则组合步数 |
| `competitor_count` | 额外 verified competing chains 数量 |
| `seed` | 随机种子 |

`conflict` 干扰不是有效规则，而是 `DECOY RULE`。prompt 明确要求忽略 DECOY；如果模型跟随它，说明筛选/抗冲突失败。

## 指标

主指标：

```text
candidate_accuracy
candidate_margin
gold_answer_ppl
```

生成诊断：

```text
generation_class = correct / conflict / competitor / distractor / miss / wrong
```

attention 诊断：

```text
gold_rule_mass_mean
non_gold_rule_mass_mean
rule_attention_selectivity
gold_rule_page_rank
```

解释规则：

1. `candidate_accuracy` 掉、`rule_attention_selectivity` 同时掉：检索/筛选退化是主要嫌疑。
2. selectivity 不低但答案错：变量绑定或中间状态维护失败更可疑。
3. chain length 增加后掉分明显，但 single-step 稳：局部组合/状态维护是瓶颈。
4. `conflict` 条件掉分明显：抗冲突筛选是瓶颈。
5. `competitor_count` 增加后掉分明显：start-code 绑定和竞争链选择是瓶颈。

## 推荐运行顺序

先跑 0.6B smoke，确认短上下文干净规则可做：

```bash
bash scripts/run_question3_boundary_smoke_server.sh
```

再跑 0.6B phase1，定位失败边界：

```bash
bash scripts/run_question3_boundary_phase1_qwen06_server.sh
```

如果 0.6B 边界清楚，再补 8B 对比：

```bash
bash scripts/run_question3_boundary_qwen8b_compare_server.sh
```

## 输出

```text
env.json
cases.jsonl
results.csv
candidate_scores.csv
attention_selectivity.csv
summary_by_condition.csv
failure_boundary.csv
summary.md
run.log
```
