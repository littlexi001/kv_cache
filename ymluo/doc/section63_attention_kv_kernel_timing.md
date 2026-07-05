# Section 63: isolated attention/KV kernel 加速测试

## 目标

前面的 warm-cache 1k decode 端到端测速显示：

```text
prompt token 大幅减少，但 1024-token decode 几乎不加速。
```

为了确认问题是不是 attention/KV kernel 本身没有加速，本节单独测试 CUDA attention kernel。

测试方式：

- 使用 `torch.nn.functional.scaled_dot_product_attention`。
- 构造 decode-like 输入：`q_len = 1`。
- 使用 Qwen3-0.6B 的真实 attention 配置：
  - layers = 28
  - attention heads = 16
  - kv heads = 8
  - head dim = 128
- dtype = fp16。
- 只测 SDPA attention 读取 K/V 的 kernel 时间，不包含 QKV projection、MLP、lm_head、Python decode loop、tokenizer、router。

脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_attention_kv_kernel_timing.py
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/attention_kv_kernel_qwen06b_20260705
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/attention_kv_kernel_qwen06b_longctx_20260705
```

## 20k 场景对应结果

对应前面单条 20k 样例：

```text
full_raw: 19455 tokens
summary1_8: 2456 tokens
retrieval_raw_k2: 4116 tokens
summary1_4: 4850 tokens
router/static_hier: 1532 tokens
```

isolated SDPA repeated-KV kernel 结果：

| KV length | avg ms / token / layer | speedup vs 19455 | 1024 tokens one layer | 1024 tokens all 28 layers |
|---:|---:|---:|---:|---:|
| 1532 | 0.029 ms | 6.72x | 30.0 ms | 0.84 s |
| 2456 | 0.034 ms | 5.75x | 35.1 ms | 0.98 s |
| 4116 | 0.052 ms | 3.82x | 52.9 ms | 1.48 s |
| 4850 | 0.059 ms | 3.34x | 60.4 ms | 1.69 s |
| 19455 | 0.197 ms | 1.00x | 202.0 ms | 5.65 s |

GQA/kv-head 口径结果非常接近：

| KV length | avg ms / token / layer | speedup vs 19455 | 1024 tokens all 28 layers |
|---:|---:|---:|---:|
| 1532 | 0.032 ms | 6.24x | 0.91 s |
| 2456 | 0.033 ms | 5.97x | 0.95 s |
| 4116 | 0.050 ms | 3.98x | 1.42 s |
| 4850 | 0.058 ms | 3.43x | 1.65 s |
| 19455 | 0.198 ms | 1.00x | 5.67 s |

## 超长上下文结果

旧 KV retrieval / KV compression 论文经常在 64k、128k 甚至更长上下文报告速度。因此额外测了 64k/128k。

| KV length | avg ms / token / layer | speedup vs 131072 |
|---:|---:|---:|
| 1532 | 0.030 ms | 41.26x |
| 19455 | 0.197 ms | 6.21x |
| 65536 | 0.623 ms | 1.97x |
| 131072 | 1.226 ms | 1.00x |

这个结果解释了为什么很多旧论文能报告很大的 kernel-level speedup：当 KV 长度到 128k 时，attention kernel 本身确实近似随 KV 长度增长；从 128k 压到 1.5k，isolated attention kernel 可以有约 `41x` 加速。

## 为什么 attention kernel 快，但整模型 warm decode 不快

对 20k 样例，isolated attention kernel 的确快：

```text
19455 KV -> 1532 KV:
  single-layer SDPA: 0.197 ms -> 0.029 ms
  speedup: 6.72x
```

但前面的 warm-cache 1k decode 是：

```text
full_raw 1024 decode: 31.461 s
router 1024 decode:   31.334 s
speedup:              1.004x
```

原因是这两个测试口径完全不同。

isolated attention kernel 不包含：

- QKV projection
- MLP
- RMSNorm
- lm_head
- sampling / argmax
- KV cache object 管理
- Python 逐 token loop
- HuggingFace model forward 调度
- kernel launch overhead

在 `Qwen3-0.6B + batch=1 + HF SDPA` 下，整模型每个 token 的固定计算和框架开销远大于“单独 attention kernel 的可节省部分”。因此 attention kernel 本身能 6x，但端到端 decode 仍然几乎不变。

## 对论文表述的影响

可以支持的说法：

```text
在 isolated attention/KV kernel 口径下，压缩历史 KV 可以带来显著加速；
20k -> 1.5k 约 6x，128k -> 1.5k 约 41x。
```

不能直接支持的说法：

```text
当前 prompt-level summary 实现可以让 HF 单请求 warm-cache 1k decode 端到端加速 6x。
```

要把 kernel-level 加速转化为端到端 serving 加速，需要：

- 真正的 compressed KV attention kernel。
- 避免 HF/Python 逐 token loop。
- 使用 vLLM / FlashInfer / TensorRT-LLM 这类 serving engine。
- 在更大模型、更长 context、更高 batch 或 decode throughput 口径下测试。
- 对 selected raw KV 做 paged KV / contiguous KV gather，而不是 prompt 文本拼接。
