# Section 62: warm-cache 后续 1k token decode 端到端测速

## 背景

前面的 `max_new_tokens=1` TTFT 测速会让 `full_raw` 重新 prefill，因此更像 cold prompt 场景。对于真正的 KV cache 服务场景，这个口径不够严格。

更合理的比较应该是：

```text
full_raw:
  原始长上下文已经 prefill，并且完整 KV cache 已经存在。
  后续推理时不重新 prefill 原始文本，只在完整 KV 上继续 decode。

summary/router:
  summary memory 或 selected memory 已经 prefill/cache。
  后续推理时在较短 KV 上继续 decode。
```

因此本节新增 warm-cache 1k decode benchmark，至少生成一个 block 长度的后续 token。

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_warm_cache_decode_timing.py
```

该脚本不用 `model.generate()` 重新处理整段 prompt，而是手动执行：

1. tokenize prompt。
2. `model(..., use_cache=True)` prefill，得到 `past_key_values`。
3. 记录 `cache_build_seconds`。
4. 使用已有 `past_key_values` 循环 decode 固定 `1024` 个 token。
5. 记录 `decode_seconds` 和 `total_seconds = cache_build_seconds + decode_seconds`。

因此这里的 `decode_seconds` 不包含重新 prefill 原始 full context。

## 实验设置

模型：

```bash
/home/fdong/hrj/prove/Qwen3-0.6B
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_warm_cache_decode1k_20260704
```

任务子集：

```text
LongBench:
  passage_count
  passage_retrieval_en

RULER:
  niah_single_1
  niah_multiquery
  cwe
  vt

RULER context length:
  8192
  16384
```

每个任务取 1 个样例，共 10 个 case。

候选方法：

```text
full_raw
summary1_8
summary1_4
retrieval_raw_k2
router
```

decode 长度：

```text
1024 tokens
```

## 总体结果

| 方法 | avg prompt tokens | token ratio | cache build | cache build speedup | 1024 decode | decode speedup | total | total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_raw | 10338 | 100.00% | 0.518s | 1.00x | 31.213s | 1.000x | 31.731s | 1.000x |
| retrieval_raw_k2 | 3536 | 34.20% | 0.143s | 3.62x | 31.098s | 1.004x | 31.241s | 1.016x |
| router | 2095 | 20.26% | 0.083s | 6.22x | 31.097s | 1.004x | 31.181s | 1.018x |
| summary1_4 | 3081 | 29.81% | 0.125s | 4.14x | 31.168s | 1.001x | 31.293s | 1.014x |
| summary1_8 | 1616 | 15.63% | 0.061s | 8.44x | 31.039s | 1.006x | 31.100s | 1.020x |

## 关键观察

这次结果和 TTFT/prefill 口径完全不同。

在 warm-cache 1k decode 口径下：

- `summary1_8` 的输入 token 只有 `15.63%`，cache build 快 `8.44x`。
- `router` 的输入 token 只有 `20.26%`，cache build 快 `6.22x`。
- 但 1024-token decode 几乎不变：`router` 只有 `1.004x`，`summary1_8` 只有 `1.006x`。
- 端到端 `cache_build + 1024 decode` 也只有约 `1.02x`。

这说明在当前 `Qwen3-0.6B + HuggingFace SDPA + Python loop` 路径下，长输出 decode 阶段不是被历史 KV 长度主导，而是被逐 token forward、MLP、框架调度、Python loop 等固定成本主导。

## 对研究 claim 的影响

这个实验修正了前面的速度理解。

可以比较有把握地说：

```text
本方法显著加速 cold prefill / TTFT / prompt-level forward。
```

但不能直接说：

```text
已有 full KV cache 后，继续生成 1k token 也能获得 4x 速度。
```

至少在当前实现上，这个 claim 不成立。warm-cache decode 的实测 speedup 只有约 `1.0x`。

## 为什么会这样

原因可能有几类：

1. 对 0.6B 小模型，逐 token decode 中 MLP 和框架 overhead 占比很高，attention over KV 不是主要瓶颈。
2. HuggingFace 逐 token loop 的 Python 调度和 kernel launch overhead 很大，KV 长度缩短后收益被淹没。
3. 当前方法是 prompt-level memory 压缩，不是真正定制的 compressed KV attention kernel。
4. decode 1024 token 时，新生成 token 本身也会不断加入 KV；随着生成变长，不同方法之间的历史 KV 长度差距会被部分稀释。

## 后续更严格的速度实验

后面如果要证明服务端 decode 加速，需要换更接近生产推理的实现：

- 使用 vLLM / TensorRT-LLM / FlashInfer 这类 optimized decode engine。
- 在大模型和更长 context 上测试，例如 Qwen3-8B，context 64k/128k。
- 报告 attention kernel 时间，而不是只报 Python 端 end-to-end。
- 实现真正的 compressed KV attention，避免 prompt 字符串压缩后重新 prefill。
- 对 retrieval raw 的 gather 做 GPU 侧连续化或 paged KV 优化。

目前最稳妥的论文表述是：

```text
我们的 adaptive summary memory 显著降低可见历史 token 和 prefill/TTFT 成本；
但 warm-cache long decode 的端到端加速需要专门的 serving/kernel 实现才能释放。
```

## 补充：单条约 20k token 样例

为了确认前面的现象不是因为上下文只有 8k/16k，又补测了一条接近 20k token 的真实 LongBench 样例。

样例：

```text
benchmark: longbench
task: gov_report
case_id: a6b66279eee0135505b08462e35738e68080a9214e36f710
full_raw prompt tokens: 19455
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_warm_cache_decode1k_20k_onecase_20260704
```

结果：

| 方法 | routed action | prompt tokens | token ratio | cache build | cache speedup | 1024 decode | decode speedup | total | total speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_raw | full_raw | 19455 | 100.00% | 1.246s | 1.00x | 31.461s | 1.000x | 32.707s | 1.000x |
| summary1_8 | summary1_8 | 2456 | 12.62% | 0.089s | 13.93x | 31.309s | 1.005x | 31.399s | 1.042x |
| summary1_4 | summary1_4 | 4850 | 24.93% | 0.193s | 6.45x | 31.432s | 1.001x | 31.625s | 1.034x |
| retrieval_raw_k2 | retrieval_raw_k2 | 4116 | 21.16% | 0.160s | 7.78x | 31.402s | 1.002x | 31.562s | 1.036x |
| router | static_hier | 1532 | 7.87% | 0.060s | 20.75x | 31.334s | 1.004x | 31.394s | 1.042x |

这个 20k 样例的结论与 10-case 子集一致：

- prompt token 降得非常明显，router 只保留 `7.87%`。
- cache build / prefill 加速非常明显，router 达到 `20.75x`。
- 但 warm-cache 后续 1024-token decode 仍然只有 `1.004x`。
- 端到端 `cache build + 1024 decode` 只有约 `1.04x`。

因此，即使把 full_raw 提高到约 20k tokens，在当前 `Qwen3-0.6B + HF SDPA + 单样本逐 token decode` 口径下，历史 KV 长度仍然不是后续 1k decode 的主要瓶颈。
