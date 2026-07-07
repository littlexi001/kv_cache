# Section 86: KV-native 与 KV-safe Action Space 小实验

## 目标

这一步验证一个关键问题：

```text
如果最终论文不想只停留在 prompt 重构，而是强调 KV cache 方法，
能不能直接从已有 KV cache 中 gather/compact 需要的历史信息，然后继续前向？
```

同时也测试一个更保守的 action space：

```text
recent 必选；
old history 由 router 根据任务难度选择：
  1. summary
  2. retrieval raw k2 / k3
  3. prefix_to_evidence
  4. evidence-local span
  5. full_old_raw fallback
```

其中 `prefix_to_evidence` 和 `span` 更接近 KV-native 安全形态，因为它们尽量保持连续 KV，而不是任意拼接非连续 page。

## 新增代码

新增 KV-native smoke：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_recent_plus_kv_native_smoke.py
```

新增 Qwen3-8B 小规模 action benchmark：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_actions_small.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/analyze_kv_safe_partial.py
```

修改 benchmark 主脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

新增 action：

```text
recent_plus_prefix_to_evidence
recent_plus_span_b0_a0
recent_plus_span_b1_a0
recent_plus_span_b1_a1
recent_plus_full_old_raw
```

## KV-native smoke 结论

在 Qwen3-0.6B 和 Qwen3-8B 上都做了真实 `past_key_values` 操作：

```text
full prefill
-> 得到真实 KV cache
-> gather selected KV pages/tokens
-> 构造 compact DynamicCache
-> query/decode 继续前向
```

核心结论：

```text
工程上可以直接 gather/compact 真实 KV cache。
但是 arbitrary sparse KV gather 不能直接等价替代 prompt-rebuild。
```

原因是：

```text
prompt-rebuild:
  selected text -> 重新 prefill -> 生成新的 hidden/K/V states

sparse KV gather:
  从原始长上下文 prefill 后的 KV cache 里抽取若干非连续 page
```

这些非连续 page 的 K/V states 是在原始长上下文位置和原始前文条件下产生的。直接拼起来以后，模型并没有被训练过这种 memory layout，因此即使命中 evidence page，也可能读不出答案。

Qwen3-8B smoke 中比较清楚：

```text
old_single:
  full_kv_cache                         exact = true
  kv_native_recent_plus_sparse_pages     exact = false
  kv_native_prefix_to_evidence_plus_recent exact = true
  prompt_rebuild_recent_plus_sparse_text exact = true

two_old:
  full_kv_cache                         exact = true
  kv_native_recent_plus_sparse_pages     exact = false
  kv_native_prefix_to_evidence_plus_recent exact = true
  prompt_rebuild_recent_plus_sparse_text exact = true
```

所以目前更稳的 KV-native 路径是：

```text
1. recent 必选
2. raw fallback 尽量使用连续 prefix/span KV
3. 任意 sparse page KV gather 需要额外 adapter 或专门训练
```

## Qwen3-8B KV-safe Action 小跑

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_actions_small_20260706
```

由于本地 ssh 超时，中断在 partial 结果；但 partial 已覆盖完整的 28 个 case，每个 case 都有全部 9 个方法，因此可以做第一版结论。

设置：

```text
model = Qwen3-8B
adapter = qwen8b_lora_4k_1ksteps_no_bench_20260705
block_tokens = 1024
recent_tokens = 512
LongBench tasks = hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count, qasper, gov_report, multi_news
RULER tasks = niah_single_1, niah_single_2, niah_multikey_1, niah_multiquery, niah_multivalue, vt, cwe, fwe
RULER lengths = 4k, 8k, 16k
```

整体结果：

| method | samples | score | full score | token ratio | seconds | relative |
|---|---:|---:|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k2 | 28 | 0.8675 | 0.7978 | 49.17% | 5.08 | 108.73% |
| recent_plus_retrieval_raw_k3 | 28 | 0.8668 | 0.7978 | 56.04% | 5.23 | 108.65% |
| full_raw | 28 | 0.7978 | 0.7978 | 100.00% | 7.09 | 100.00% |
| recent_plus_full_old_raw | 28 | 0.7965 | 0.7978 | 100.12% | 7.09 | 99.83% |
| recent_plus_span_b0_a0 | 28 | 0.5817 | 0.7978 | 25.06% | 4.50 | 72.92% |
| recent_plus_prefix_to_evidence | 28 | 0.5475 | 0.7978 | 38.00% | 4.84 | 68.62% |
| recent_plus_span_b1_a1 | 28 | 0.5461 | 0.7978 | 45.18% | 4.94 | 68.45% |
| recent_plus_span_b1_a0 | 28 | 0.5460 | 0.7978 | 30.35% | 4.66 | 68.44% |
| recent_plus_summary1_8 | 28 | 0.4736 | 0.7978 | 19.48% | 4.48 | 59.36% |

按 benchmark 切分：

| benchmark | full_raw | retrieval_k2 | retrieval_k3 | prefix_to_evidence | span_b0_a0 | span_b1_a0 | span_b1_a1 | summary1_8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LongBench | 0.2924 | 0.5361 | 0.5338 | 0.2911 | 0.4111 | 0.2861 | 0.1613 | 0.2825 |
| RULER 4k | 1.0000 | 1.0000 | 1.0000 | 0.6250 | 0.6250 | 0.6250 | 0.7500 | 0.7500 |
| RULER 8k | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.3750 |
| RULER 16k | 1.0000 | 1.0000 | 1.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.5000 |

## 解释

这组结果说明两件事。

第一，当前 prompt-level 的 `recent_plus_retrieval_raw_k2/k3` 仍然是质量最强的低 token 方法：

```text
k2: 108.73% full_raw score, 49.17% tokens
k3: 108.65% full_raw score, 56.04% tokens
```

这说明 recent + 少量 evidence raw block 的方向本身是有效的。

第二，直接改成更 KV-safe 的 `prefix_to_evidence` 或局部 `span` 后，质量明显下降：

```text
prefix_to_evidence: 68.62% full_raw
span_b0_a0:         72.92% full_raw
span_b1_a0:         68.44% full_raw
span_b1_a1:         68.45% full_raw
```

主要原因不是 KV-safe 这个想法错，而是当前证据选择太粗糙：

```text
1. 现在只用 query lexical overlap 打分 block。
2. 多证据任务可能需要多个 block，但 span/prefix 当前只围绕 top1 evidence。
3. RULER 16k 下，如果 top1 block 错了，prefix/span 会直接失败。
4. prefix_to_evidence 为了保持连续性，会包含从开头到 evidence 的所有 old tokens；
   如果 evidence 选错，token 花了但信息不对。
```

## 当前判断

当前已经验证了三个层次：

```text
1. 方法方向有效：
   recent_plus_retrieval_raw_k2/k3 在小样本上超过 full_raw，且 token 明显更少。

2. KV-native 工程路径可行：
   可以对真实 past_key_values 做 gather/compact 并继续 decode。

3. 任意 sparse KV gather 不是免费替代：
   原模型对非连续 KV page 拼接不鲁棒，需要 adapter 或更保守的连续 KV action。
```

因此下一步最合理的路线不是继续堆普通 benchmark，而是把 action/router 做成更强的两阶段系统：

```text
stage 1: risk / difficulty router
  判断是否需要 raw old history，是否需要多证据，是否需要全文 fallback。

stage 2: KV-safe selector
  在低风险时用 summary 或 retrieval k2/k3；
  在需要 KV-native raw fallback 时，选择 prefix/span/full_old；
  对多证据任务选择 top-k span 或 prefix_to_farthest_topk；
  高风险直接 full_old_raw。
```

更具体地，下一轮应该测试：

```text
recent_plus_span_top2_b0_a0
recent_plus_span_top2_b1_a0
recent_plus_prefix_to_farthest_top2
recent_plus_prefix_to_farthest_top3
risk_fallback_full_old
```

这样能把当前最强的 retrieval 质量和 KV-native 安全约束接起来。

