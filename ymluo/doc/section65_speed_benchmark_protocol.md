# Section 65: KV cache / memory compression 速度基准口径规范

## 目标

本文档定义后续 KV cache、memory compression、summary memory、retrieval/paged KV、router-adaptive 方法的统一测速口径。

核心原则：

```text
不同方法按真实执行路径计时；
共同成本不要混入 attention/KV 子系统结论；
方法新增开销必须计入，并做多步、多 query 摊销。
```

不要只用 HuggingFace `generate()` 的端到端时间作为唯一速度结论。`generate()` 会混入 MLP、lm_head、tokenizer、Python loop、HF forward 调度、kernel launch 等大量所有方法共有的开销。它可以作为最终参考指标，但不能替代 attention/KV 子系统速度分析。

## 必须分开的测速口径

后续速度实验至少要分开报告以下指标。

### 1. Cold prefill / TTFT

长 prompt 从零输入模型，到生成第一个 token 的时间。

适合说明：

```text
长上下文重新输入模型时，多久开始输出。
```

注意：

- 这个口径允许 full_raw 重新 prefill 原文。
- 但不能把这个结果误当成 warm-cache decode 加速。

### 2. Cache build / compression time

构建 full KV、summary KV、compressed KV、retrieval index、page metadata、router feature 等的时间。

如果某些结构可以离线构建或历史阶段构建，需要单独报告：

```text
shared build cost
per-query online cost
multi-query amortized cost
```

### 3. Warm-cache decode

历史 KV 已经存在后，继续生成固定长度 token，例如 1024 tokens。

这个口径下：

```text
full_raw baseline 不应该重新 prefill 原始 prompt。
```

full_raw 应该直接使用已有完整 `past_key_values`，然后继续 decode。

### 4. Attention/KV subsystem

只测与 KV 长度、KV 选择、KV gather、attention over KV 直接相关的时间。

应该包含每个方法真实引入的 attention/KV 子系统开销。

不应该包含所有方法共有、且基本与历史 KV 长度无关的成本：

- MLP
- lm_head
- tokenizer
- 完整 HuggingFace `generate()`
- Python 逐 token decode loop
- 完整 transformer block 中与 KV 长度无关的部分

### 5. Amortized new-query overhead

新 query/prompt 到来时的一次性开销，需要摊到多步 decode 上。

至少报告：

```text
steps = 1, 16, 64, 256, 1024
```

对应：

- overhead time
- attention time
- total subsystem time
- overhead share
- speedup vs full attention

## Full Raw Baseline

### Cold prefill 场景

full_raw 可以重新输入完整 prompt，并记录：

- prompt tokens
- prefill time
- TTFT
- peak KV memory

### Warm-cache 场景

full_raw 必须使用已有完整 KV cache。

应记录：

- full_raw cache build time
- full_raw warm-cache decode time
- full_raw attention/KV subsystem time

不能把 `full_raw` 重新 prefill 原文的时间和压缩方法的 warm decode 时间直接比较。

## 不同方法的计时规则

不同优化方法不能强行套同一组额外开销。每个方法只计算它实际执行的操作。

### A. Full Raw

需要计入：

- attention 到完整 KV cache
- cold prefill 场景下的完整 prompt prefill
- warm-cache decode 场景下的完整 KV attention

不需要计入：

- router
- page scoring
- top-k selection
- KV gather / compact
- summary generation

### B. Fixed Summary Memory

必须区分 summary 是离线构建还是 query-time 构建。

如果 summary 是历史阶段或离线阶段已经构建好的：

- 单独报告 summary generation / summary KV build time
- query-time 只计入 summary memory 读取、summary KV attention
- warm-cache decode 不应每步重新生成 summary

如果 summary 是 query 到来后动态生成的：

- 必须把 summary generation time 计入该 query 的开销
- 必须把 summary KV build / tokenize / prefill 计入
- 再计入后续多步 attention 到 summary KV

需要报告：

- summary compression ratio
- summary build time
- summary read/load time
- attention 到 summary KV 的时间
- 多 query 摊销后的平均成本

### C. Retrieval / Paged KV

需要计入：

- query-aware block/page scoring
- retrieval index lookup
- top-k page/block selection
- KV page gather / copy / compact，或 paged attention 的索引访问开销
- 后续多步 attention 到 selected KV / paged KV

如果 retrieval index 是离线构建：

- index build time 单独报告
- query-time 只报告 lookup / scoring / top-k / gather

如果每隔 N 步重新选择 page：

- 不应默认每步重选，除非方法确实这样做
- 应测试 interval，例如 128、512、1024 steps
- 每次重选开销需要摊销到后续 decode steps

需要报告：

- page size
- selected pages 数量
- selected KV length
- scoring time
- top-k time
- gather / compact time
- paged attention time
- interval-based amortized speedup

### D. Router-Adaptive Policy

router 本身只是策略选择，不代表每个 action 都有同样开销。

需要计入：

- router feature extraction time
- router forward time
- router 选择出的 action 的真实开销

具体规则：

```text
router -> full_raw:
  router overhead + full KV attention
  不计入 gather 或 summary generation

router -> retrieval:
  router overhead + retrieval scoring/top-k/gather + selected KV attention

router -> summary:
  router overhead + summary memory 读取或构建 + summary KV attention

router -> fallback full attention:
  router overhead + full KV attention
```

需要报告：

- router overhead
- routed action distribution
- 每种 action 的平均 latency
- overall latency
- oracle policy vs distilled router policy 的速度和准确率差异

### E. Learned / Generative Summarizer

如果 summarizer 是小模型或 teacher model，需要单独报告：

- summarizer inference time
- summary token 数
- summary KV build time
- summarizer 是否离线运行
- summarizer 是否 query-dependent

如果 summary 是 query-independent：

- 可以作为历史压缩成本
- 应报告多 query amortized cost

如果 summary 是 query-dependent：

- 必须计入每个 query 的 latency
- 不能把它当成免费 memory

## Attention/KV Subsystem 推荐表格

推荐使用真实模型配置，单独测 attention/KV 子系统。

full_attention：

```text
多步 attention 到完整 KV。
```

my_method：

```text
新 query 到来时，执行该方法真实需要的 router / scoring / top-k / gather / summary read。
然后多步 attention 到 compact KV / selected KV / summary KV。
```

推荐表头：

```text
method
full_kv_len
active_kv_len
page_size
selected_pages
steps
router_time
scoring_time
topk_time
gather_time
summary_read_time
attention_time
total_time
overhead_share
speedup_vs_full_attention
```

至少报告：

```text
steps = 1, 16, 64, 256, 1024
```

## Warm-Cache Decode 推荐流程

### Full Raw

```text
1. prefill 历史上下文，得到完整 past_key_values。
2. 不重新 prefill 原文。
3. 使用完整 past_key_values 继续生成 1024 tokens。
```

### 压缩方法

```text
1. 构建 compressed / summary / selected KV。
2. 使用 compressed KV 继续生成 1024 tokens。
3. 单独记录 compression / selection / gather / cache build time。
```

同时报告：

- cache build time
- compression time
- query-time overhead
- 1k decode time
- total first-query latency
- multi-query amortized latency

## 多 Query 摊销

如果某些成本可以被多个 query 共享，例如：

- summary memory build
- retrieval index build
- page metadata build
- compressed KV construction

必须报告多 query 摊销。

推荐：

```text
Q = 1, 4, 16, 64
```

公式：

```text
amortized_latency = query_time_latency + shared_build_time / Q
```

需要报告：

- Q=1 latency
- Q=4 latency
- Q=16 latency
- Q=64 latency
- shared build cost
- per-query cost

## Claim 边界

可以说：

- 本方法降低可见 KV/token 数。
- 本方法加速 cold prefill / TTFT。
- 本方法在 attention/KV subsystem 上有加速。
- 本方法新增的 router / retrieval / gather / summary 读取开销可以通过多步 decode 或多 query 摊销。
- 在 optimized serving/kernel 实现下，attention/KV subsystem 加速有潜力转化为 end-to-end throughput 加速。

不要直接说：

- token 少，完整 HuggingFace `generate()` 就一定同比例变快。
- prompt-level 压缩一定能让 warm-cache 1k decode 端到端加速 4x。
- full_raw 在 warm-cache baseline 里应该重新 prefill 原文。
- 所有方法都应该计入同样的 router/gather/summary generation 开销。

## End-to-End Decode 不快时的解释

如果 HuggingFace end-to-end warm decode speedup 很小，应这样解释：

```text
KV 压缩主要减少 attention over history 的成本。
但小模型、batch=1、Python loop、HF forward、MLP、lm_head、kernel launch 等共同成本可能占主导。
因此 attention/KV subsystem 可能有明显加速，但完整 end-to-end decode latency 不一定明显变快。
这不代表方法没有优化 KV/attention，而是说明需要 serving/kernel 级实现才能释放端到端收益。
```

## 最终实验包

后续论文或报告至少需要三类结果。

### A. Accuracy / Quality

比较：

- full_raw
- fixed summary
- retrieval / paged KV
- router-adaptive
- oracle policy
- distilled router

### B. Token / KV Reduction

报告：

- active KV length
- compression ratio
- selected pages
- summary tokens
- routed action distribution

### C. Speed

报告：

- cold prefill / TTFT
- cache build / compression time
- attention/KV subsystem speed
- new-query overhead amortized speed
- warm-cache 1k decode end-to-end speed
- multi-query amortized latency

结论中必须明确每个 speedup 对应的口径，不要把 kernel-level speedup、attention-subsystem speedup、TTFT speedup、end-to-end generate speedup 混为一谈。
