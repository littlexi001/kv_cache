# 第 46 节：学习式分层摘要记忆实验

日期：2026-07-03

## 0. 实验目标

本节验证一个和第 45 节不同的方向：

```text
不是用 mean-K summary 找 raw KV，
而是训练一个小 summarizer，
把每 1w raw tokens 压成 10 / 100 / 1000 token 三层知识记忆。
```

query 来的时候，直接选择对应粒度的 summary memory 前向：

```text
全局主题 / block 主题:
  用 10-token summary

人物角色 / block outcome:
  用 100-token summary

事件 action / result:
  用 1000-token summary

精确 verification code:
  才 fallback 到 raw tokens
```

这更接近用户提出的“像人整理知识一样压缩上下文”的方向。

## 1. 脚本

新增脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_learned_hier_summary_memory.py
```

输出目录：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/default
```

主要输出：

```text
summary.csv
by_kind.csv
trials.csv
summary.json
examples.jsonl
```

## 2. Synthetic 设置

每本 synthetic book 包含多个 10k-token block。每个 block 有四层信息：

```text
10-token summary:
  theme / setting / conflict

100-token summary:
  outcome / character roles

1000-token summary:
  event action / event result

raw tokens:
  exact verification code
```

小 summarizer 是一个 slot-aware MLP：

```text
global encoder:
  整个 raw block -> theme / setting / conflict / outcome

role encoder:
  对每个人物的证据片段 -> role

event encoder:
  对每个事件 slot 的证据片段 -> action / result
```

这里的设计重点不是追求真实自然语言生成，而是验证：

```text
学习出来的 summary memory 自身能否承载 query 所需知识。
```

## 3. 主要结果

| 方法 | Accuracy | 平均 token cost | 相对 raw cost |
| --- | ---: | ---: | ---: |
| full_raw | 100.00% | 16000 | 100.00% |
| gold_adaptive_no_raw | 86.67% | 436 | 2.73% |
| gold_adaptive_with_raw | 100.00% | 1636 | 10.23% |
| learned_adaptive_no_raw | 85.00% | 436 | 2.73% |
| learned_adaptive_with_raw | 98.33% | 1636 | 10.23% |
| learned_summary1000_only | 85.00% | 1600 | 10.00% |
| learned_summary100_only | 58.33% | 160 | 1.00% |
| learned_summary10_only | 32.17% | 16 | 0.10% |

关键结果：

```text
learned_adaptive_no_raw:
  accuracy = 85.00%
  cost = 2.73% raw

learned_adaptive_with_raw:
  accuracy = 98.33%
  cost = 10.23% raw
```

这说明：对于非精确细节题，大多数 query 可以直接由 learned summary memory 回答；raw tokens 主要用于 exact code 这种无法压缩的精确细节。

## 4. 分任务结果

`learned_adaptive_with_raw`：

| 任务 | Accuracy | 使用层级 |
| --- | ---: | --- |
| block_theme | 96.25% | summary10 |
| block_conflict | 97.50% | summary10 |
| book_theme | 95.00% | summary10 across blocks |
| block_outcome | 96.25% | summary100 |
| character_role | 100.00% | summary100 |
| event_action | 100.00% | summary1000 |
| event_result | 100.00% | summary1000 |
| exact_code | 100.00% | raw fallback |

小 summarizer 的槽位准确率：

```text
theme_acc = 95.25%
setting_acc = 97.00%
conflict_acc = 96.50%
outcome_acc = 96.50%
active_role_acc = 100.00%
event_action_acc = 100.00%
event_result_acc = 100.00%
```

## 5. 当前结论

这个实验支持用户修正后的方向：

```text
1. summary 不应该只是 raw token retrieval 的索引。
2. 学习式 summary memory 本身可以作为前向上下文。
3. 不同 query 应该使用不同粒度的 memory。
4. raw token 应该作为精确细节 fallback，而不是默认路径。
```

和第 45 节 mean-K KV routing 的区别：

```text
第 45 节:
  summary = mean K
  作用 = 找 raw KV
  最终 attention = raw KV

第 46 节:
  summary = 小模型学出的知识记忆
  作用 = 直接回答 / 直接作为前向上下文
  raw = 只在 exact query fallback
```

## 6. 下一步

后续应该把 synthetic slot-aware summarizer 换成更真实的小模型：

```text
1. 用小 Transformer / T5-style summarizer 生成 10/100/1000 token 文本摘要。
2. 用 Qwen3 读取这些摘要文本，测真实 QA / PPL。
3. 训练 query router 判断使用 10、100、1000 还是 raw。
4. 在长书、LongBench、RULER 上测试。
```
