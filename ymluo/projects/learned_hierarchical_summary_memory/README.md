# Learned Hierarchical Summary Memory

这个实验验证一种和 mean-K KV routing 不同的方向：

```text
每 1w raw tokens -> 小 summarizer -> 10 / 100 / 1000 token 三层知识记忆
query 来时直接用对应层级的 summary memory 前向
只有精确细节题才回退 raw tokens
```

这里的 summary 不是 K-cache 均值，也不是 raw KV 的索引。它是小模型从 raw block 中学出来的结构化知识。

## 实验设计

每个 synthetic book 由多个 10k-token block 组成。每个 block 有：

- 10-token 级别知识：主题、场景、核心冲突。
- 100-token 级别知识：结局、人物角色。
- 1000-token 级别知识：事件链，包含每个事件的 action/result。
- raw 级别知识：精确 verification code。

训练一个 slot-aware 小 MLP summarizer：

```text
global encoder: raw block text -> theme / setting / conflict / outcome
role encoder: character evidence span -> role
event encoder: event evidence span -> action / result
```

然后比较：

- `full_raw`
- `gold_adaptive_no_raw`
- `gold_adaptive_with_raw`
- `learned_summary10_only`
- `learned_summary100_only`
- `learned_summary1000_only`
- `learned_adaptive_no_raw`
- `learned_adaptive_with_raw`

## 运行

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_learned_hier_summary_memory.py
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/default/
  summary.csv
  by_kind.csv
  trials.csv
  summary.json
  examples.jsonl
```

## 解释

如果 `learned_adaptive_with_raw` 在大幅低于 raw token cost 的情况下接近 `full_raw`，说明：

```text
不同 query 确实应该使用不同粒度的 learned summary memory。
summary 本身可以作为前向上下文，而不只是 raw token retrieval 的索引。
raw tokens 应该作为精确细节 fallback，而不是默认路径。
```

默认结果：

```text
full_raw:
  accuracy = 100.00%
  cost = 100.00% raw

learned_adaptive_no_raw:
  accuracy = 85.00%
  cost = 2.73% raw

learned_adaptive_with_raw:
  accuracy = 98.33%
  cost = 10.23% raw
```
