# 第 47 节：真实文本与经典 KV Retrieval Benchmark 的摘要记忆测试

日期：2026-07-03

## 0. 目标

在第 46 节 synthetic learned summary memory 之后，本节补两类测试：

```text
1. 真实文本：
   用 War and Peace 文本做 extractive 分层摘要记忆测试。

2. 经典 KV retrieval benchmark：
   passkey / needle / KV lookup / conflict latest / multihop / exact code。
```

注意：本节仍然是轻量评估框架，不是最终真实 LLM 端到端实验。

```text
真实文本部分：
  用 TF-IDF / entity / sentence extraction 生成 10/100/1000 摘要。

classic KV 部分：
  用受控构造的 summary memory 测不同 query 应该落在哪个记忆层级。
```

## 1. 新增脚本

真实文本：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_real_text_summary_memory_eval.py
```

经典 KV retrieval：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_classic_kv_retrieval_summary_benchmark.py
```

服务器已运行：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_warpeace_20260703
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/classic_kv_retrieval_20260703
```

本地输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_warpeace_20260703
ymluo/projects/learned_hierarchical_summary_memory/outputs/classic_kv_retrieval_20260703
```

## 2. 真实文本：War and Peace

文本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt
```

设置：

```text
max_words = 80000
block_words = 10000
blocks = 8
summary10_tokens = 10
summary100_tokens = 100
summary1000_tokens = 1000
raw_block_tokens = 10000
```

任务类型：

| 任务 | 含义 | 预期层级 |
| --- | --- | --- |
| book_keyword | 整本文本的高频主题词 | summary10 across blocks |
| block_keyword | 某个 block 的关键词 | summary10 |
| entity | 某个 block 的主要实体 | summary100 |
| exact_sentence | 精确句子回忆 | raw fallback |

### 结果

| 方法 | Accuracy | 平均 token cost | 相对 raw cost |
| --- | ---: | ---: | ---: |
| full_raw | 100.00% | 12800.0 | 100.00% |
| adaptive_no_raw | 68.00% | 358.4 | 2.80% |
| adaptive_with_raw | 100.00% | 3238.4 | 25.30% |
| summary1000_only | 68.00% | 1280.0 | 10.00% |
| summary100_only | 68.00% | 128.0 | 1.00% |
| summary10_only | 36.00% | 12.8 | 0.10% |

分任务看：

```text
block_keyword:
  adaptive_no_raw = 100%
  adaptive_with_raw = 100%

book_keyword:
  adaptive_no_raw = 100%
  adaptive_with_raw = 100%

entity:
  adaptive_no_raw = 100%
  adaptive_with_raw = 100%

exact_sentence:
  adaptive_no_raw = 0%
  adaptive_with_raw = 100%
```

解释：

```text
真实文本中的关键词和实体类问题可以由摘要记忆回答。
精确句子回忆不能指望摘要完整保留，必须 raw fallback。
```

## 3. Classic KV Retrieval Benchmark

任务类型：

| Variant | 含义 | 预期层级 |
| --- | --- | --- |
| passkey | 找隐藏 passkey | summary100 |
| needle | needle-in-haystack 标签查找 | summary100 |
| kv_lookup | key -> label 表查找 | summary1000 |
| conflict_latest | 当前值覆盖旧值 | summary100 |
| multihop | project -> artifact -> action | summary1000 |
| exact_code | 精确 code | raw fallback |

设置：

```text
tasks_per_variant = 80
variants = 6
total_tasks = 480
distractor_records = 96
raw_context_tokens = 10000
```

### 总体结果

| 方法 | Accuracy | 平均 token cost | 相对 raw cost |
| --- | ---: | ---: | ---: |
| full_raw | 100.00% | 10000.0 | 100.00% |
| adaptive_no_raw | 83.33% | 550.0 | 5.50% |
| adaptive_with_raw | 100.00% | 2050.0 | 20.50% |
| summary1000_only | 83.33% | 1000.0 | 10.00% |
| summary100_only | 66.67% | 100.0 | 1.00% |
| summary10_only | 0.00% | 10.0 | 0.10% |

分任务看：

```text
adaptive_no_raw:
  passkey = 100%
  needle = 100%
  conflict_latest = 100%
  kv_lookup = 100%
  multihop = 100%
  exact_code = 0%

adaptive_with_raw:
  所有 variant = 100%
```

解释：

```text
经典 retrieval 任务大多可以被结构化 summary memory 覆盖。
exact_code 这类精确字符串仍然必须 raw fallback。
```

## 4. 当前结论

这两组测试支持一个更清楚的 memory policy：

```text
1. summary memory 应该是默认路径。
2. query 类型决定使用 10 / 100 / 1000 哪一层。
3. raw token 不应该默认参与前向。
4. raw token 主要服务于 exact quote / exact code / 不可压缩细节。
```

和第 45 节的 mean-K routing 相比，这里的重点已经变成：

```text
summary 本身就是知识载体，
而不是 raw KV retrieval index。
```

## 5. 下一步

下一步应该上真实 LLM 端到端版本：

```text
1. 用小模型或 Qwen 自己为每 1w tokens 生成 10/100/1000 token 摘要。
2. 对真实 QA prompt，只喂对应层级 summary，让 Qwen 回答。
3. 对 exact quote / exact code 类 query 才拼 raw block。
4. 对比 full raw context、summary-only、adaptive-summary、adaptive+raw fallback。
5. 在 RULER / NIAH / passkey / LongBench 子集上报告准确率和 token cost。
```
