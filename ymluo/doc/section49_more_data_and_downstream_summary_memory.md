# 第 49 节：更多数据与下游任务测试

日期：2026-07-03

## 0. 目标

第 48 节已经验证了 summary memory 在 classic KV retrieval 上的 token cost 和 prefill 时间优势。本节继续补两类测试：

```text
1. 更多真实文本数据：在 Count of Monte Cristo 上复测真实文本 summary memory。
2. 更多下游任务：新增 10 类 controlled downstream task，并同时跑 symbolic oracle 和 Qwen3-0.6B 真实生成。
```

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_extended_summary_memory_downstream.py
```

这个脚本生成显式的：

```text
raw_context
summary10
summary100
summary1000
query
answer
```

因此同一批样例既可以做 symbolic oracle 测试，也可以直接喂给 Qwen3-0.6B 生成答案。

## 2. 新增真实文本：Count of Monte Cristo

文本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_montecristo_20260703_route
```

本地输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_montecristo_20260703_route
```

结果：

| 方法 | tasks | accuracy | avg token cost | raw cost ratio |
| --- | ---: | ---: | ---: | ---: |
| full_raw | 25 | 100.00% | 12800.0 | 100.00% |
| adaptive_no_raw | 25 | 68.00% | 358.4 | 2.80% |
| adaptive_with_raw | 25 | 100.00% | 3238.4 | 25.30% |
| summary1000_only | 25 | 68.00% | 1280.0 | 10.00% |
| summary100_only | 25 | 68.00% | 128.0 | 1.00% |
| summary10_only | 25 | 36.00% | 12.8 | 0.10% |

这个结果和 War and Peace 一致：关键词/实体类可以由 summary memory 回答，exact sentence 类必须 raw fallback。

## 3. Extended Controlled Downstream Suite

### 3.1 任务类型

新增 10 类任务：

| task type | 含义 | 预期 memory level |
| --- | --- | --- |
| topic_summary | 判断长块的主主题 | summary10 |
| passkey | 隐藏 passkey 查找 | summary100 |
| needle | key -> label needle 查找 | summary100 |
| conflict_latest | old/current 冲突中取 current | summary100 |
| variable_tracking | 多步变量更新后取 final value | summary100 |
| kv_lookup | 表格 key -> label 查找 | summary1000 |
| multihop | project -> artifact -> action | summary1000 |
| multiquery | 两个 key 同时查两个 label | summary1000 |
| aggregation_count | 统计某类 label 的数量 | summary1000 |
| exact_code | 精确 verification code | raw fallback |

### 3.2 Symbolic oracle 结果

设置：

```text
tasks_per_variant = 50
variants = 10
total_tasks = 500
distractor_records = 128
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/extended_downstream_symbolic_20260703
```

结果：

| 方法 | tasks | accuracy | avg token cost | raw cost ratio |
| --- | ---: | ---: | ---: | ---: |
| full_raw | 500 | 100.00% | 10000.0 | 100.00% |
| adaptive_no_raw | 500 | 90.00% | 541.0 | 5.41% |
| adaptive_with_raw | 500 | 100.00% | 1441.0 | 14.41% |
| summary1000_only | 500 | 90.00% | 1000.0 | 10.00% |
| summary100_only | 500 | 62.60% | 100.0 | 1.00% |
| summary10_only | 500 | 10.00% | 10.0 | 0.10% |

路由占比：

| 方法 | summary10 | summary100 | summary1000 | full attention |
| --- | ---: | ---: | ---: | ---: |
| adaptive_no_raw | 10.00% | 40.00% | 50.00% | 0.00% |
| adaptive_with_raw | 10.00% | 40.00% | 40.00% | 10.00% |

解释：

```text
adaptive_no_raw 唯一失败来自 exact_code。
adaptive_with_raw 只在 exact_code 上回退 raw，因此恢复到 100%。
```

## 4. Qwen3-0.6B 真实生成测试

### 4.1 设置

服务器：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
GPU = RTX 3090
dtype = float16
attention = sdpa
```

数据：

```text
tasks_per_variant = 5
variants = 10
total_tasks = 50
distractor_records = 64
max_new_tokens = 32
methods = full_raw, summary1000_only, adaptive_no_raw, adaptive_with_raw
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/extended_downstream_qwen3_5shot_v2_20260703
```

本地输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/extended_downstream_qwen3_5shot_v2_20260703
```

### 4.2 总体结果

| 方法 | tasks | accuracy | avg token cost | raw cost ratio | avg input tokens | avg seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_raw | 50 | 58.00% | 10000.0 | 100.00% | 1801.8 | 1.067 |
| adaptive_no_raw | 50 | 62.00% | 541.0 | 5.41% | 218.0 | 0.987 |
| adaptive_with_raw | 50 | 72.00% | 1441.0 | 14.41% | 422.1 | 0.991 |
| summary1000_only | 50 | 54.00% | 1000.0 | 10.00% | 220.3 | 0.995 |

这里最重要的现象是：

```text
adaptive_with_raw 的真实生成准确率高于 full_raw，同时只用了 14.41% 名义 raw token cost。
```

full_raw 不是上界。对 Qwen3-0.6B 这种小模型，长 raw context 里的 distractor 会明显干扰生成。

### 4.3 分任务结果

| task type | full_raw | adaptive_no_raw | adaptive_with_raw | summary1000_only |
| --- | ---: | ---: | ---: | ---: |
| topic_summary | 60% | 100% | 100% | 100% |
| passkey | 100% | 100% | 100% | 60% |
| needle | 100% | 100% | 100% | 100% |
| conflict_latest | 60% | 80% | 80% | 40% |
| variable_tracking | 60% | 100% | 100% | 100% |
| aggregation_count | 20% | 100% | 100% | 100% |
| exact_code | 100% | 0% | 100% | 0% |
| multihop | 80% | 40% | 40% | 40% |
| kv_lookup | 0% | 0% | 0% | 0% |
| multiquery | 0% | 0% | 0% | 0% |

## 5. 结论

这组测试让结论更清楚：

```text
1. adaptive_with_raw 在 controlled oracle 中保持 100% accuracy，token cost 只有 14.41% raw。
2. 在 Qwen3-0.6B 真实生成中，adaptive_with_raw 仍高于 full_raw：72% vs 58%。
3. summary memory 不只是省 token，有时还能减少 raw distractor 对小模型的干扰。
4. exact_code 仍然必须 raw fallback；no_raw 在这类任务上必然失败。
```

但也暴露了边界：

```text
1. kv_lookup 和 multiquery 在 Qwen3-0.6B 上仍然失败。
   原因不是 memory level 选错，而是 summary1000 里仍是一张较大的 key-value 表，小模型不稳定会查错。

2. multihop 也没有完全稳定。
   这说明 summary1000 不能只是“压缩后的长表/长记录”，最好进一步变成 query-conditioned evidence。
```

所以后续更合理的方向是：

```text
raw block -> learned static summaries:
  summary10 / summary100 / summary1000

query + summaries -> small planner / extractor:
  选 memory level
  从 summary1000 中抽出 query-specific evidence
  只有 exact quote / exact code 才打开 raw block
```

这比“summary1000 直接完整喂给 LLM”更接近真正的人类记忆检索。

