# Section 87: KV-safe Top-k Action 结果

## 目标

上一版 `prefix_to_evidence` 和 `span_b*_a*` 只围绕 top1 evidence block 做选择，质量偏低：

```text
prefix_to_evidence: 约 68.6% full_raw
span_b0_a0:         约 72.9% full_raw
```

这说明问题不在 recent-plus 方向本身，而在于：

```text
1. top1 evidence 经常不够；
2. 多证据任务需要多个 block；
3. KV-native 更安全的连续 prefix/span action 需要 top-k 证据选择。
```

因此这一步加入：

```text
recent_plus_span_top2_b0_a0
recent_plus_span_top2_b1_a0
recent_plus_span_top3_b0_a0
recent_plus_prefix_to_farthest_top2
recent_plus_prefix_to_farthest_top3
recent_plus_kv_safe_rule_v0
```

## 新增代码

修改：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_topk_actions_small.sh
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_topk_actions_small_20260707
```

## 设置

```text
model = Qwen3-8B
adapter = qwen8b_lora_4k_1ksteps_no_bench_20260705
block_tokens = 1024
recent_tokens = 512
max_examples_per_task = 1

LongBench:
  hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count,
  qasper, gov_report, multi_news

RULER:
  niah_single_1, niah_single_2, niah_multikey_1, niah_multiquery,
  niah_multivalue, vt, cwe, fwe

RULER lengths:
  4k, 8k, 16k
```

总计：

```text
32 cases
10 methods
320 trials
```

注意：

```text
RULER 16k / cwe 的 full_raw、full_old_raw、kv_safe_rule_v0 出现 OOM。
因此主表采用 full_raw 成功的 31 个 case 做公平统计。
```

## 公平统计结果

只统计 `full_raw` 成功的 31 个 case：

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k2 | 0.8803 | 0.8174 | 107.70% | 46.19% | 4.97 |
| recent_plus_retrieval_raw_k3 | 0.8797 | 0.8174 | 107.62% | 52.60% | 5.16 |
| recent_plus_span_top3_b0_a0 | 0.8480 | 0.8174 | 103.75% | 40.63% | 4.87 |
| full_raw | 0.8174 | 0.8174 | 100.00% | 100.00% | 7.37 |
| recent_plus_prefix_to_farthest_top3 | 0.8171 | 0.8174 | 99.96% | 58.49% | 5.52 |
| recent_plus_full_old_raw | 0.8162 | 0.8174 | 99.85% | 100.11% | 7.30 |
| recent_plus_span_top2_b0_a0 | 0.8158 | 0.8174 | 99.80% | 34.85% | 4.70 |
| recent_plus_prefix_to_farthest_top2 | 0.7848 | 0.8174 | 96.01% | 54.35% | 5.41 |
| recent_plus_kv_safe_rule_v0 | 0.7837 | 0.8174 | 95.88% | 65.73% | 5.95 |
| recent_plus_span_top2_b1_a0 | 0.7835 | 0.8174 | 95.85% | 42.88% | 4.91 |

## 按 benchmark 切分

| benchmark | full_raw | retrieval_k2 | retrieval_k3 | span_top3_b0_a0 | prefix_farthest_top3 | span_top2_b0_a0 |
|---|---:|---:|---:|---:|---:|---:|
| LongBench | 0.2924 | 0.5361 | 0.5338 | 0.4111 | 0.2911 | 0.4111 |
| RULER 4k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULER 8k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8750 |
| RULER 16k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

对应 token ratio：

| benchmark | retrieval_k2 | retrieval_k3 | span_top3_b0_a0 | prefix_farthest_top3 | span_top2_b0_a0 |
|---|---:|---:|---:|---:|---:|
| LongBench | 47.8% | 57.7% | 38.2% | 57.6% | 30.8% |
| RULER 4k | 75.3% | 79.5% | 67.3% | 77.5% | 63.1% |
| RULER 8k | 38.5% | 46.1% | 35.2% | 52.2% | 27.6% |
| RULER 16k | 19.9% | 23.6% | 19.2% | 45.0% | 15.5% |

## 关键观察

### 1. top-k span 明显修复了上一版 top1 span 的问题

上一版 `span_b0_a0` 约为：

```text
relative = 72.9% full_raw
token ratio = 25.1%
```

这版 `span_top3_b0_a0`：

```text
relative = 103.75% full_raw
token ratio = 40.63%
```

这说明多证据选择是必须的。只选 top1 block 太脆弱，尤其在 RULER multi-key / multi-query / multi-value 和 LongBench 多跳 QA 上会漏证据。

### 2. `span_top3_b0_a0` 是目前最有价值的 KV-safe 候选

它的性质是：

```text
只选 top3 evidence block；
不额外扩展 before/after；
recent raw 必选；
token ratio 约 40.6%；
score 超过 full_raw；
速度约 4.87s vs full_raw 7.37s。
```

虽然严格来说多个 span 仍然可能是非连续的，但它比 arbitrary retrieval prompt 更接近可转成 KV page selection 的形态。

### 3. prefix-to-farthest top3 很稳，但 token 偏高

`prefix_to_farthest_top3` 的质量几乎等于 full_raw：

```text
relative = 99.96% full_raw
token ratio = 58.49%
```

它的优点是 KV 结构更安全，因为可以保留从开头到最远证据的连续 old prefix。

缺点是如果最远证据很靠后，就会吃掉很多 old tokens。

适合做高风险 fallback，而不适合作为默认策略。

### 4. retrieval_raw_k2/k3 仍然是质量最强 baseline

公平口径：

```text
retrieval_raw_k2: relative = 107.70%, token = 46.19%
retrieval_raw_k3: relative = 107.62%, token = 52.60%
```

这说明“recent + 少量 old evidence raw”方向非常强。

但它当前是 prompt-level 方法，不是最保守的 KV-native 形态。论文里需要清楚地区分：

```text
prompt-rebuild / training proxy
KV-native page/span selection
```

### 5. rule_v0 不够好

`recent_plus_kv_safe_rule_v0`：

```text
relative = 95.88%
token ratio = 65.73%
```

它的问题是太保守且规则不精细：

```text
1. 有些 RULER task 被路由到 full_old_raw，导致 token 高，甚至 16k cwe OOM；
2. 有些 multi-evidence case 应该用 span_top3，却只用了较弱策略；
3. summary / exact / retrieval / global aggregation 没有分得足够细。
```

所以 rule_v0 只能当 smoke baseline，不能作为最终 router。

## 当前结论

这一步结果是正面的：

```text
best prompt-level quality:
  recent_plus_retrieval_raw_k2
  107.70% full_raw, 46.19% tokens

best KV-safe-ish candidate:
  recent_plus_span_top3_b0_a0
  103.75% full_raw, 40.63% tokens

safe high-risk fallback:
  recent_plus_prefix_to_farthest_top3
  99.96% full_raw, 58.49% tokens
```

相比上一版，top-k evidence selection 已经把 KV-safe action 从明显不可用提升到接近或超过 full_raw。

## 下一步

下一步不应该继续手写规则，而应该蒸馏一个 router：

```text
label space:
  recent_plus_summary1_8
  recent_plus_retrieval_raw_k2
  recent_plus_span_top2_b0_a0
  recent_plus_span_top3_b0_a0
  recent_plus_prefix_to_farthest_top3
  recent_plus_full_old_raw

训练数据:
  非 benchmark synthetic retrieval / multi-evidence / summary / global aggregation
  普通文本 PPL 样本
  长度覆盖 4k / 8k / 16k / 32k

目标:
  让 router 学会：
    简单 exact -> span_top2 或 span_top3
    多证据 exact -> span_top3
    高风险全文任务 -> prefix_to_farthest_top3 或 full_old_raw
    总结任务 -> summary ratio
    prompt-level 允许时 -> retrieval_raw_k2
```

如果 router 能接近 oracle，在当前实验形态下，主张会比之前更完整：

```text
质量: 接近或超过 full_raw
token: 约 20%-45%
速度: attention/prompt 侧有实测收益
工程: KV-native path 有 smoke demo
边界: sparse KV gather 不等价，需 adapter 或 KV-safe action
```

