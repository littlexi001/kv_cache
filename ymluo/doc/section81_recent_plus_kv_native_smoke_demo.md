# Section 81: Recent-plus KV-native Smoke Demo

## 目标

本节回答一个关键问题：

```text
现在的方法是不是还只是 prompt 重组？
能不能直接使用已有 KV cache，gather/compact 选中的 KV page，然后继续前向？
```

结论先说：

```text
工程上可以直接 gather/compact 真实 Qwen KV cache 并继续 decode。
但是 prompt-rebuild 和 arbitrary sparse KV gather 不是等价操作。
原模型没有自然学会从任意拼接的非连续 KV page 里读答案。
更稳的 KV-native 形态是 contiguous prefix/span KV，或者训练 gathered-KV adapter。
```

## 脚本

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_recent_plus_kv_native_smoke.py
```

它比较以下方法：

```text
full_kv_cache
kv_native_recent_only
kv_native_recent_plus_sparse_pages_absolute_pos
kv_native_recent_plus_sparse_pages_compact_pos
kv_native_prefix_to_evidence_plus_recent
prompt_rebuild_recent_plus_sparse_text
```

其中：

```text
full_kv_cache:
  prefill 完整上下文，query 直接 attend 完整 KV。

kv_native_recent_only:
  从 full KV 里只 gather 最近 recent tokens。

kv_native_recent_plus_sparse_pages_*:
  从 full KV 里 gather 证据 old page + recent KV。
  不重新 prefill 文本。

kv_native_prefix_to_evidence_plus_recent:
  gather 从开头到证据页的连续 prefix/span KV，再加 recent KV。
  这是更保守、更符合原模型 causal KV 结构的 fallback。

prompt_rebuild_recent_plus_sparse_text:
  把证据 old page 文本 + recent 文本重新拼成 prompt，再 prefill。
  这是当前质量 benchmark 主要使用的近似方式。
```

## 0.6B Smoke

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_kv_native_smoke_qwen06b_8k_20260706
```

设置：

```text
model = Qwen3-0.6B
context = 8192 tokens
page = 1024 tokens
recent = 512 tokens
decode_steps = 24
```

### old_single

答案在 old page 里。

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 8192 | 1.00x | 0.9839 | true |
| kv_native_recent_only | 512 | 1.08x | 12.3398 | false |
| kv_native_recent_plus_sparse absolute | 1536 | 1.08x | 6.9993 | false |
| kv_native_recent_plus_sparse compact | 1536 | 1.08x | 7.3325 | false |
| kv_native_prefix_to_evidence_plus_recent | 6656 | 1.07x | 3.6700 | false |
| prompt_rebuild_recent_plus_sparse_text | 1502 | 1.05x | 0.9614 | true |

观察：

```text
prompt-rebuild 可以答对；
直接 sparse KV gather 失败；
prefix/span 更好但 0.6B 仍不稳定。
```

### recent_single

答案在 recent 512 token 里。

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 8192 | 1.00x | 0.9237 | true |
| kv_native_recent_only | 512 | 1.01x | 8.0581 | false |
| kv_native_recent_plus_sparse absolute | 512 | 1.02x | 8.0581 | false |
| kv_native_recent_plus_sparse compact | 512 | 1.01x | 6.3342 | false |
| prompt_rebuild_recent_plus_sparse_text | 520 | 1.02x | 1.8785 | true |

观察：

```text
即使答案在 recent KV 里，直接拿 recent KV 给 query attend 也不等价于把 recent text 放到 prompt 里重新 prefill。
这说明 query 和 selected memory 的交互方式很重要。
```

### two_old

答案需要两个 old page 的 bridge evidence。

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 8192 | 1.00x | 0.8897 | true |
| kv_native_recent_only | 512 | 1.01x | 12.1059 | false |
| kv_native_recent_plus_sparse absolute | 2560 | 1.00x | 10.5022 | false |
| kv_native_recent_plus_sparse compact | 2560 | 1.00x | 8.1772 | false |
| kv_native_prefix_to_evidence_plus_recent | 7680 | 1.00x | 0.9803 | true |
| prompt_rebuild_recent_plus_sparse_text | 2486 | 0.93x | 1.0460 | true |

观察：

```text
two-hop 下，连续 prefix/span KV 可以答对；
arbitrary sparse page KV 仍然失败；
prompt-rebuild 也答对，但需要重新 prefill。
```

## 8B Smoke

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_kv_native_smoke_qwen8b_6k_20260706
```

设置：

```text
model = Qwen3-8B
context = 6144 tokens
page = 1024 tokens
recent = 512 tokens
decode_steps = 16
```

### old_single

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 6144 | 1.00x | 0.8930 | true |
| kv_native_recent_only | 512 | 1.14x | 7.3318 | false |
| kv_native_recent_plus_sparse absolute | 1536 | 1.14x | 3.0082 | false |
| kv_native_recent_plus_sparse compact | 1536 | 1.14x | 2.0101 | false |
| kv_native_prefix_to_evidence_plus_recent | 5632 | 1.12x | 0.8813 | true |
| prompt_rebuild_recent_plus_sparse_text | 1502 | 0.76x | 2.1802 | true |

观察：

```text
8B 上 prefix-to-evidence + recent KV 可以答对，并且 NLL 接近 full。
sparse KV gather 仍然没有 exact，虽然已经能生成部分答案片段。
```

### recent_single

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 6144 | 1.00x | 0.9549 | true |
| kv_native_recent_only | 512 | 1.02x | 2.1790 | false |
| kv_native_recent_plus_sparse absolute | 512 | 1.02x | 2.1790 | false |
| kv_native_recent_plus_sparse compact | 512 | 1.02x | 2.5438 | false |
| prompt_rebuild_recent_plus_sparse_text | 521 | 0.88x | 0.9224 | true |

观察：

```text
8B 的 recent-only KV 比 0.6B 好很多，但仍没有 exact。
prompt-rebuild 可以答对。
```

### two_old

| method | active KV | speedup | NLL | exact |
|---|---:|---:|---:|---|
| full_kv_cache | 6144 | 1.00x | 1.4632 | true |
| kv_native_recent_only | 512 | 1.02x | 7.9258 | false |
| kv_native_recent_plus_sparse absolute | 2560 | 1.02x | 3.3264 | false |
| kv_native_recent_plus_sparse compact | 2560 | 1.02x | 4.6801 | false |
| kv_native_prefix_to_evidence_plus_recent | 5632 | 1.00x | 1.5373 | true |
| prompt_rebuild_recent_plus_sparse_text | 2486 | 0.53x | 1.5559 | true |

观察：

```text
prefix-to-evidence + recent KV 与 full KV 质量接近；
prompt-rebuild 也答对但在线慢；
sparse KV gather 不稳定。
```

## 关键解释

### 为什么 prompt-rebuild 能答对，而 sparse KV gather 会失败？

prompt-rebuild 做的是：

```text
selected text -> 重新 prefill -> 新的 hidden states / K/V states
```

模型在这个新 prompt 里重新建立了 query 前的上下文结构，所以答案容易被读出来。

KV gather 做的是：

```text
full context prefill 后，从原始 KV cache 里抽取某些 page 的 K/V states
```

这些 K/V states 是在原始长上下文的位置和前文条件下计算出来的。把非连续 page 直接拼起来后，模型没有训练过这种 memory layout，因此 query 不一定能像读 prompt 一样读出答案。

所以：

```text
prompt-rebuild recent-plus != KV-native sparse gather recent-plus
```

这是非常重要的技术边界。

## 当前结论

1. **KV-native 工程路径是可行的。**

   已经可以：

   ```text
   full prefill -> 得到真实 past_key_values
   gather selected KV tokens/pages
   构造 compact DynamicCache
   query/decode 继续前向
   ```

2. **任意稀疏 page KV gather 不能直接替代 prompt-rebuild。**

   即使 evidence page 被选中，原模型也可能读不出答案。

3. **连续 prefix/span KV 是当前最安全的 raw KV fallback。**

   Qwen3-8B 上：

   ```text
   old_single: prefix_to_evidence_plus_recent exact = true
   two_old:    prefix_to_evidence_plus_recent exact = true
   ```

4. **prompt-rebuild 仍然是好的质量 oracle / training proxy，但不是最终工程形态。**

   它证明“选这些文本信息足够回答”，但不能证明“直接 gather 这些 KV page 也能回答”。

## 对当前方法的影响

论文方法应该避免声称：

```text
任意 top-k raw KV page gather 可以无损替代 prompt 重构。
```

更稳的表述是：

```text
router 选择 memory resolution；
summary memory 可以作为压缩主路径；
raw KV fallback 应该使用 KV-safe layouts，例如 continuous prefix/span/recent；
arbitrary sparse KV page gather 需要专门 adapter 训练或 paged-attention 形式的模型适配。
```

## 下一步建议

### 1. KV-safe action space

把 raw KV action 从：

```text
retrieval_raw_k2/k3/k4 arbitrary pages
```

改成：

```text
retrieval_span_k1/k2
prefix_to_evidence
recent_plus_prefix_span
summary_plus_span
```

### 2. Gathered-KV adapter

如果仍想保留 arbitrary sparse page KV，需要训练 LoRA adapter 适应：

```text
[gathered old KV pages] + [recent KV] + query
```

训练数据可以使用非 benchmark synthetic exact/retrieval，目标是让模型在 evidence page 已选中的情况下能读出答案。

### 3. Paper 叙事

现在可以清楚区分：

```text
prompt-rebuild experiments:
  验证 policy / memory content 是否足够。

KV-native smoke:
  验证能否直接操作 KV cache；
  同时揭示 arbitrary sparse KV layout 的风险。

attention/KV timing:
  验证如果 active KV layout 可用，速度收益足够大。
```

这比单纯说“我们不是 RAG”更有说服力。
