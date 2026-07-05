# Section 56: 全文信息压力测试与 raw fallback 的必要性

## 背景

前面的实验说明，`[summary memory] + [recent raw]` 在普通 next-token prediction 上可以显著降低输入 token，并且经过 LoRA 适配后，PPL 还能保持得比较好。但是这个结论不能直接外推到所有长上下文任务。

一个关键问题是：summary 是有损压缩。只要任务需要逐字、逐事实、全文证据，例如精确查询某个编号、原文引用、计数、顺序比较、多处证据核对，summary-only 就可能把答案本身压掉。这个风险不能只靠平均 PPL 发现，因为 PPL 更偏向语言连续性，不等价于 exact evidence recall。

因此这里补了一个专门的全文信息压力测试。

## 实验脚本

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_full_information_stress_eval.py
```

实验构造方式：

1. 在真实 public-domain 文本中取 8k token prefix。
2. 在较早的历史 block 中插入一条 synthetic exact memory record，例如：

```text
SPECIAL MEMORY RECORD.
The private access code for MOBY-000-524 is lumen-391-cipher.
This value must be recalled exactly.
```

3. 查询：

```text
Question: What is the private access code for MOBY-000-524?
Answer with only the code.
```

4. 用四选一 target scoring 测试模型是否选择正确答案。

指标：

- `answer_retained_rate`：正确答案字符串是否还出现在压缩后的 prompt 中。这是信息保留的硬上限。
- `choice_accuracy`：模型在四个候选答案中是否选对。
- `token_ratio_vs_full_raw`：输入 token 相对 full raw 的比例。
- `speedup_vs_full_raw`：当前 PyTorch/SDPA 实测 forward 加速。

对比方法：

- `full_raw`：完整原始上下文。
- `learned_static_hier`：learned extractive summary + static hierarchical routing + recent raw。
- `learned_static_sum1000`：每个旧 block 使用较长 summary1000。
- `retrieval_raw_k1`：summary memory + query-aware raw block retrieval top-1。
- `retrieval_raw_k2`：summary memory + query-aware raw block retrieval top-2。
- `risk_gate_full_on_exact`：检测到 exact-evidence 查询时直接回退 full raw。

## Base Qwen3-0.6B 结果

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/full_information_stress_base_4texts_s4_20260703
```

数据：`moby_dick`, `pride_prejudice`, `origin_species`, `republic`，每本 4 个样例，共 16 个 exact key-value 样例。

| 方法 | answer retained | choice accuracy | tokens vs full | forward speedup |
|---|---:|---:|---:|---:|
| full_raw | 100.00% | 100.00% | 100.00% | 1.00x |
| learned_static_hier | 0.00% | 18.75% | 23.53% | 5.19x |
| learned_static_sum1000 | 0.00% | 18.75% | 61.28% | 1.88x |
| retrieval_raw_k1 | 100.00% | 100.00% | 48.43% | 2.40x |
| retrieval_raw_k2 | 100.00% | 100.00% | 68.62% | 1.63x |
| risk_gate_full_on_exact | 100.00% | 100.00% | 100.00% | 1.01x |

结论非常直接：summary-only 的答案保留率是 0。它虽然输入最短、速度最快，但是对于 exact evidence recall，本质上已经把答案删除了。四选一准确率只有 18.75%，接近随机，不能认为它真正完成了任务。

`retrieval_raw_k1` 只取一个 query-relevant raw block，就恢复了 100% answer retention 和 100% choice accuracy，同时输入 token 仍然只有 full raw 的约 48.43%，实测 forward 约 2.40x 加速。

## LoRA adapted model 结果

adapter：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s1000_20260703/adapter
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/full_information_stress_lora_longbooks_s4_20260703
```

| 方法 | answer retained | choice accuracy | tokens vs full | forward speedup |
|---|---:|---:|---:|---:|
| full_raw | 100.00% | 100.00% | 100.00% | 1.00x |
| learned_static_hier | 0.00% | 25.00% | 23.53% | 4.80x |
| learned_static_sum1000 | 0.00% | 25.00% | 61.28% | 1.81x |
| retrieval_raw_k1 | 100.00% | 100.00% | 48.43% | 2.29x |
| retrieval_raw_k2 | 100.00% | 100.00% | 68.62% | 1.58x |
| risk_gate_full_on_exact | 100.00% | 100.00% | 100.00% | 1.01x |

LoRA 适配没有改变本质结论。adapter 可以让模型更适应 summary memory 的输入格式，但如果正确答案已经被 summary 删除，adapter 不可能凭空恢复 exact raw 信息。

## 对当前方法的影响

这个实验说明，目前最好的表述不应该是：

> 用 summary memory 替代原始 KV cache。

更准确的表述应该是：

> 用 summary memory 作为低成本语义记忆；对于需要精确证据的 query，通过 raw retrieval 或 full raw fallback 恢复原始信息。

也就是说，`adaptive_no_raw` 只能作为 fast semantic mode，适合概括、主题延续、普通生成和不要求逐字证据的场景。它不适合作为默认通用模式。

更实际的默认系统应当是：

```text
summary memory always on
+ recent raw always on
+ query-aware raw retrieval when exact evidence is needed
+ full raw fallback when risk is high or retrieval confidence is low
```

## 推荐的下一版方法

### 1. 保留 raw block，但不默认放进 GPU attention

原始 token 或原始 KV 不应该完全丢弃，而是放在 CPU memory、host memory、磁盘、paged KV 或压缩 raw block store 中。GPU 上默认只跑 summary memory；当 query 需要精确证据时，再 gather 少量 raw block。

这和本次实验里的 `retrieval_raw_k1` 很一致：只取一个 raw block，速度比 full raw 慢一些，但仍然明显快于 full raw，并且能恢复精确信息。

### 2. 增加 task/query-aware router

router 至少应该区分三类 query：

| query 类型 | 推荐 memory |
|---|---|
| 概括、主题理解、普通续写 | summary-only 或 summary + recent raw |
| 指定实体、编号、quote、exact answer | summary + raw retrieval |
| 计数、全局排序、全文核对、多处证据 | 多 raw block retrieval 或 full raw fallback |

目前可以先用 heuristic gate 做原型，例如检测 `exact`, `quote`, `code`, `which value`, `where`, `when`, `how many`, `list all` 等高风险模式。后续可以训练一个 router，用验证集上的 answer retention / downstream accuracy 做监督信号。

### 3. 训练目标要加入 exact-evidence stress set

只用 next-token PPL 训练 adapter 会偏向语言连续性。下一版训练和验证应该混入：

- synthetic exact key-value recall；
- needle-in-a-haystack；
- quote retrieval；
- multi-needle retrieval；
- count / list / order 类任务；
- LongBench / RULER 中需要证据定位的子任务。

validation-based early stopping 也要同时看 PPL 和 exact-evidence accuracy，不能只看 PPL。

## 目前结论

用户的担心是正确的：如果方法压缩掉 raw 信息，那么对于需要全文信息的任务会产生很糟糕的结果。本节实验已经把这个失败模式量化出来了。

但这个失败并不否定 summary memory 的价值。它说明方法边界必须改成 hybrid memory：

- summary memory 提供低成本语义上下文；
- raw retrieval 提供精确证据；
- risk gate 决定何时回退；
- 对不需要精确证据的 query 保持高压缩和高速度。

从论文角度看，这个结果反而很重要：它明确了 no-raw summary memory 的不可行边界，也给出了一个可工作的 practical system design。
