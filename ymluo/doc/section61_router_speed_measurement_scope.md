# Section 61: router 速度口径复查

## 背景

上一节 runtime router 在完整生成口径下只有 `1.19x` speedup，看起来明显低于之前 PPL/prefill 实验里的 `4x+`。

复查后确认：这两个数字不是同一个测速口径。

## 完整生成口径

上一节的 runtime router 使用：

```bash
--max_new_tokens_exact 48
--max_new_tokens_summary 160
```

在这组任务里，exact 类任务实际平均生成 48 个 token。这个口径测的是：

```text
prompt 构造 + tokenize + prefill + 48-token decode + Python/HF generate overhead
```

因此即使输入 token 从 100% 降到 32.46%，总耗时也只从 full raw 的约 `1.91s` 降到 router 的约 `1.62s`，最终 speedup 只有约 `1.19x`。

已有 48-token 生成结果：

| 方法 | 平均输入 token | token ratio | 平均生成耗时 | speedup |
|---|---:|---:|---:|---:|
| full_raw | 9779 | 100.00% | 1.914s | 1.00x |
| router | 3155 | 32.46% | 1.619s | 1.18x |
| summary1_8 | 1527 | 15.61% | 1.514s | 1.26x |
| static_hier | 1077 | 11.01% | 1.496s | 1.28x |

这里可以看到，即使是 `static_hier`，输入只有约 11%，但总耗时仍有 1.50s。说明固定 decode 和框架开销占比很大，把 prefill 的收益稀释了。

## 1-token 近似 prefill/TTFT 口径

为了和之前 PPL/prefill 更公平地对齐，重新跑了一次：

```bash
--max_new_tokens_exact 1
--max_new_tokens_summary 1
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_prefill_timing_1tok_20260704
```

总体结果：

| 方法 | 平均输入 token | token ratio | 平均耗时 | speedup |
|---|---:|---:|---:|---:|
| full_raw | 9779 | 100.00% | 0.445s | 1.00x |
| router | 3155 | 32.27% | 0.127s | 3.52x |
| retrieval_raw_k2 | 3515 | 35.95% | 0.121s | 3.69x |
| summary1_4 | 2939 | 30.05% | 0.103s | 4.32x |
| summary1_8 | 1527 | 15.61% | 0.054s | 8.24x |
| static_hier | 1077 | 11.01% | 0.040s | 11.05x |
| summary1_2 | 5717 | 58.47% | 0.224s | 1.99x |
| summary1000 | 9165 | 93.72% | 0.407s | 1.09x |

这个结果说明：方法本身在长上下文 prefill/TTFT 上确实能达到甚至超过 `4x`。router 由于会为了准确率选择 `retrieval_raw_k2`、`summary1000`、`full_raw` 等较贵动作，所以整体是 `3.52x`，略低于固定 `summary1_4` 的 `4.32x`，但准确率会更好。

## 结论

`1.19x` 不是方法没有速度潜力，而是完整生成口径下 decode 和框架固定开销太大。

对论文或报告，建议分开报告：

- `Prefill/TTFT speedup`：这是长上下文压缩方法最应该报告的核心速度，当前 router 是 `3.52x`，`summary1_4` 是 `4.32x`。
- `End-to-end generation speedup`：包含 decode，当前 48-token 生成只有 `1.19x`。
- `Input token reduction`：router 是约 `32% full_raw`。

如果要让完整生成也接近 4x，需要做工程侧优化：

- 使用真正的 compressed KV cache，而不是 prompt-level 压缩后重新 tokenize。
- 避免 Python 字符串拼接和 HuggingFace `generate()` 的 per-sample overhead。
- batch 化 router 和 prefill。
- 对 retrieval raw 的 gather 做 CUDA/kernel 级优化，避免 Python 侧动态拼接 raw block。
- 在长输出任务里，结合 speculative decoding 或 decode cache 复用，否则 decode 阶段本身不会因为历史压缩而同比例变快。
