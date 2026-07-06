# Section 75: Recent-required Old-memory Policy 快速实验

## 目标

本节测试新的问题定义：

```text
recent raw 永远保留；
router 只决定 old context 用什么粒度回忆。
```

也就是不再把 `recent_only` 当作一个独立 action，而是把所有 action 改成：

```text
recent raw + old summary
recent raw + old retrieval
recent raw + old full raw
```

这样更符合生成任务的直觉：

```text
recent raw 负责当前生成状态和局部连贯性；
old memory 负责回答前文知识问题。
```

并且假设：

```text
当前 query 的答案在 old context 中；
recent 是必选上下文，不是 router 选择对象。
```

## 代码改动

在 `run_qwen8b_paper_benchmarks.py` 中新增 action：

```text
recent_plus_summary1_8
recent_plus_summary1_4
recent_plus_summary1_2
recent_plus_static_hier
recent_plus_retrieval_raw_k1
recent_plus_retrieval_raw_k2
recent_plus_retrieval_raw_k3
recent_plus_retrieval_raw_k4
recent_plus_retrieval_raw_k8
recent_plus_full_old_raw
```

实现逻辑：

```text
context = old context + recent context

old context = context 去掉最后 recent_tokens
recent context = 最后 recent_tokens

memory =
  Old memory:
    selected old-memory representation

  Recent raw context:
    recent raw tokens
```

在 `run_synthetic_router_distillation.py` 中也支持了 `recent_plus_*` 的 synthetic success 判断。

## Qwen3-0.6B 真实生成 Quick Test

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_policy_qwen06b_quick_20260705
```

配置：

```text
model = Qwen3-0.6B
LongBench tasks = hotpotqa, passage_retrieval_en, gov_report, multi_news
RULER tasks = niah_single_1, niah_multiquery, vt, cwe
context length = RULER 8192
max_examples_per_task = 1
```

这个实验很小，只看方向。

### 方法整体结果

| method | score | token ratio |
|---|---:|---:|
| full_raw | 0.5303 | 100.00% |
| recent_only | 0.1468 | 7.62% |
| summary1_8 | 0.1358 | 12.54% |
| retrieval_raw_k1 | 0.5092 | 35.17% |
| retrieval_raw_k2 | 0.7750 | 46.27% |
| recent_plus_summary1_8 | 0.1375 | 18.87% |
| recent_plus_static_hier | 0.0099 | 17.34% |
| recent_plus_retrieval_raw_k1 | 0.6250 | 31.84% |
| recent_plus_retrieval_raw_k2 | 0.5257 | 42.95% |

### Oracle 对比

在这 8 个样例上重算 strict best-score oracle：

| action space | relative to full | token ratio | 主要选择 |
|---|---:|---:|---|
| old_action_space | 147.23% | 33.96% | retrieval_raw_k2, full_raw, recent_only |
| recent_required_space | 123.74% | 34.12% | recent_plus_retrieval_k1/k2, full_raw |
| recent_required_no_full | 122.70% | 35.61% | recent_plus_retrieval_k1/k2 |

解释：

- 旧 action space 里仍然会选 `recent_only`。
- 新 action space 去掉了 `recent_only`，oracle 不再被“答案碰巧在 recent 或 full_raw 失败”污染。
- 新 action space 在 RULER old-answer 样例上表现干净：多数选择 `recent_plus_retrieval_raw_k1/k2`。

### RULER 8192 old-answer 观察

在 RULER 样例上：

| method | score | token ratio |
|---|---:|---:|
| full_raw | 0.7500 | 100.00% |
| recent_plus_retrieval_raw_k1 | 1.0000 | 26.40% |
| recent_plus_retrieval_raw_k2 | 1.0000 | 34.34% |

这说明：

```text
在答案确实来自 old memory 的 exact/retrieval 场景中，
recent required + old retrieval 是有效的。
```

## Controlled Synthetic: 答案在 Old Context

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_old_answer_synthetic_20260705
```

配置：

```text
cases_per_dataset = 60
length = 8192
datasets = War and Peace, Monte Cristo
candidate actions = full_raw + recent_plus_*
```

### Router 结果

| split | success | token ratio |
|---|---:|---:|
| train | 100.00% | 51.26% |
| test | 87.10% | 54.39% |

这个结果不是最终 benchmark，只说明新 action space 在 controlled old-answer 数据上是可学的。

### Oracle Label 分布

| oracle label | count |
|---|---:|
| recent_plus_retrieval_raw_k1 | 31 |
| recent_plus_retrieval_raw_k3 | 30 |
| recent_plus_retrieval_raw_k4 | 18 |
| full_raw | 16 |
| recent_plus_summary1_8 | 12 |
| recent_plus_summary1_4 | 6 |
| recent_plus_retrieval_raw_k2 | 5 |
| recent_plus_retrieval_raw_k8 | 2 |

### 按任务类型看

| kind | oracle pattern |
|---|---|
| magic_single_old | recent_plus_retrieval_raw_k1 |
| single_old | recent_plus_retrieval_raw_k1 |
| two_old | recent_plus_retrieval_raw_k2 |
| three_old | recent_plus_retrieval_raw_k3 |
| four_old | recent_plus_retrieval_raw_k4 |
| magic_multiquery | recent_plus_retrieval_raw_k3 |
| magic_multivalue | recent_plus_retrieval_raw_k4 |
| summary_brief | recent_plus_summary1_8 |
| summary_detailed | recent_plus_summary1_4 |

这个 pattern 比旧 oracle 更符合直觉：

```text
单证据 -> k1
两证据 -> k2
三证据 -> k3
四证据 -> k4
简单总结 -> summary1_8
详细总结 -> summary1_4
```

## 当前判断

这个新设定是更合理的。

相比旧设定：

```text
旧设定:
  router 可以直接选 recent_only
  oracle 容易被 recent_only 污染

新设定:
  recent raw 固定保留
  router 只决定 old memory 粒度
  action label 更接近“任务难度”
```

但也有一个代价：

```text
recent 永远保留会增加最低 token ratio。
```

在 controlled synthetic 里，oracle token ratio 大约是：

```text
51.5%
```

这比之前 18%-20% 的 oracle 高很多。原因是这个 synthetic 里多证据 old-answer 样例较多，而且 recent 固定加入成本。

所以后续需要同时报告两个 token ratio：

```text
total token ratio = (recent + selected old memory) / full raw
old-memory token ratio = selected old memory / old raw
```

否则 recent 固定成本会让压缩率看起来变差。

## 下一步

下一步建议：

1. 用 Qwen3-8B 正式跑一版小 benchmark：

   ```text
   full_raw
   recent_plus_summary1_8
   recent_plus_summary1_4
   recent_plus_retrieval_raw_k1
   recent_plus_retrieval_raw_k2
   recent_plus_retrieval_raw_k4
   recent_plus_retrieval_raw_k8
   ```

2. 重新定义 router label：

   ```text
   fixed recent
   old_memory_action in {summary ratios, retrieval k, full_old_raw}
   ```

3. 训练 router 时不要再出现 `recent_only` label。

4. 评估时单独统计：

   ```text
   quality vs full_raw
   total token ratio
   old-memory token ratio
   selected old action 分布
   ```

## 简短结论

这次 quick test 支持你的判断：

```text
recent 应该固定保留；
router 应该只选择 old memory 的回忆粒度。
```

这个 formulation 比旧的 `recent_only` action space 更干净，也更适合写成论文方法。
