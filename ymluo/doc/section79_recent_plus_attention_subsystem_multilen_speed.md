# Section 79: Recent-plus Attention/KV 子系统多长度速度测试

## 目标

本节测试当前 recent-plus 方法在 attention/KV 子系统里的速度。

测速口径：

```text
只测 attention/KV subsystem。
计入本方法额外引入的 router / page scoring / top-k / KV gather / compact。
不计入 tokenizer、MLP、lm_head、完整 HF generate、采样等共同成本。
```

也就是说，这个结果回答的是：

```text
如果 KV cache 已经存在，
新 query 到来后先做路由和 KV page compact，
然后继续 decode 多步，
attention/KV 部分能加速多少。
```

## 脚本

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_recent_plus_attention_subsystem_timing.py
```

分析脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/analyze_recent_plus_attention_timing.py
```

服务器输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_attention_subsystem_qwen8b_multilen_warm_20260706
```

原始结果：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_attention_subsystem_qwen8b_multilen_warm_20260706/recent_plus_attention_subsystem_timing.csv
```

## 实验设置

模型形状来自 Qwen3-8B：

```text
layers = 36
query heads = 32
KV heads = 8
head_dim = 128
dtype = fp16
batch = 1
```

测试长度：

```text
4096, 8192, 16384, 20000, 32768
```

decode steps：

```text
1, 64, 256, 1024
```

recent-plus 设置：

```text
page_size = 1024
recent_tokens = 512

k2 active KV = 2 * 1024 + 512 = 2560
k3 active KV = 3 * 1024 + 512 = 3584
k4 active KV = 4 * 1024 + 512 = 4608
```

注意：

```text
当前服务器 PyTorch SDPA native_gqa = false。
脚本用 repeat_interleave KV fallback 来跑 GQA。
因此绝对时间偏保守，但 full attention 和 recent-plus compact attention 是同一口径，speedup 可以比较。
```

## 1024 步主结果

### page_once

新 query 到来时只做一次：

```text
router + scoring + top-k + gather/compact
```

后续 1024 token 都 attention 到 compact KV。

| full length | method | active KV | active ratio | overhead | attention | total | speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | full_attention | 4096 | 100.00% | 0.000 ms | 9374.826 ms | 9374.826 ms | 1.00x |
| 4096 | recent_plus_k2 | 2560 | 62.50% | 2.439 ms | 6625.524 ms | 6627.963 ms | 1.41x |
| 4096 | recent_plus_k3 | 3584 | 87.50% | 3.350 ms | 8871.688 ms | 8875.038 ms | 1.06x |
| 4096 | recent_plus_k4 | 3584 | 87.50% | 3.415 ms | 8787.676 ms | 8791.091 ms | 1.07x |
| 8192 | full_attention | 8192 | 100.00% | 0.000 ms | 17393.395 ms | 17393.395 ms | 1.00x |
| 8192 | recent_plus_k2 | 2560 | 31.25% | 2.471 ms | 6731.038 ms | 6733.509 ms | 2.58x |
| 8192 | recent_plus_k3 | 3584 | 43.75% | 3.356 ms | 8619.126 ms | 8622.482 ms | 2.02x |
| 8192 | recent_plus_k4 | 4608 | 56.25% | 4.364 ms | 10558.729 ms | 10563.094 ms | 1.65x |
| 16384 | full_attention | 16384 | 100.00% | 0.000 ms | 38986.121 ms | 38986.121 ms | 1.00x |
| 16384 | recent_plus_k2 | 2560 | 15.62% | 2.431 ms | 6750.980 ms | 6753.411 ms | 5.77x |
| 16384 | recent_plus_k3 | 3584 | 21.88% | 3.368 ms | 8661.500 ms | 8664.868 ms | 4.50x |
| 16384 | recent_plus_k4 | 4608 | 28.12% | 4.362 ms | 10559.469 ms | 10563.831 ms | 3.69x |
| 20000 | full_attention | 20000 | 100.00% | 0.000 ms | 47277.023 ms | 47277.023 ms | 1.00x |
| 20000 | recent_plus_k2 | 2560 | 12.80% | 2.457 ms | 6801.022 ms | 6803.479 ms | 6.95x |
| 20000 | recent_plus_k3 | 3584 | 17.92% | 3.378 ms | 8702.350 ms | 8705.728 ms | 5.43x |
| 20000 | recent_plus_k4 | 4608 | 23.04% | 4.393 ms | 10582.348 ms | 10586.741 ms | 4.47x |
| 32768 | full_attention | 32768 | 100.00% | 0.000 ms | 76292.320 ms | 76292.320 ms | 1.00x |
| 32768 | recent_plus_k2 | 2560 | 7.81% | 2.444 ms | 6805.567 ms | 6808.012 ms | 11.21x |
| 32768 | recent_plus_k3 | 3584 | 10.94% | 3.372 ms | 8719.400 ms | 8722.772 ms | 8.75x |
| 32768 | recent_plus_k4 | 4608 | 14.06% | 4.398 ms | 10651.683 ms | 10656.081 ms | 7.16x |

## 20k 长度的关键结论

对于一条 20k 长度数据，生成 1024 token：

| method | active KV | overhead | total | speedup |
|---|---:|---:|---:|---:|
| full_attention | 20000 | 0.000 ms | 47277.023 ms | 1.00x |
| recent_plus_k2 | 2560 | 2.457 ms | 6803.479 ms | 6.95x |
| recent_plus_k3 | 3584 | 3.378 ms | 8705.728 ms | 5.43x |
| recent_plus_k4 | 4608 | 4.393 ms | 10586.741 ms | 4.47x |

这说明：

```text
在 attention/KV 子系统口径下，
当前方法的额外开销不是瓶颈。
真正决定速度的是 active KV 长度。
```

## 额外开销分解

1024 步、page_once 下，不同 k 的一次性开销平均值：

| method | router + scoring + top-k | gather + compact | total overhead |
|---|---:|---:|---:|
| recent_plus_k2 | 0.315 ms | 2.134 ms | 2.448 ms |
| recent_plus_k3 | 0.294 ms | 3.070 ms | 3.365 ms |
| recent_plus_k4 | 0.289 ms | 3.897 ms | 4.187 ms |

解释：

```text
router / scoring / top-k 基本只有 0.3 ms。
主要额外开销是跨 layer 的 KV gather/compact。
但 gather/compact 也只有 2-4 ms 量级。
摊到 1024 步时，overhead share 约 0.04%。
```

## 1 步新 query 场景

如果只生成 1 token，额外开销占比很明显。

20k 长度：

| method | overhead | attention | total | speedup |
|---|---:|---:|---:|---:|
| full_attention | 0.000 ms | 44.828 ms | 44.828 ms | 1.00x |
| recent_plus_k2 | 2.457 ms | 5.662 ms | 8.118 ms | 5.52x |
| recent_plus_k3 | 3.378 ms | 7.432 ms | 10.810 ms | 4.15x |
| recent_plus_k4 | 4.393 ms | 9.588 ms | 13.981 ms | 3.21x |

即使只生成 1 token，20k 以上上下文仍然有明显收益。但在 4k 短上下文里，k3/k4 active ratio 已经接近 full，所以 overhead 会让它们不划算。

## 每 128 步重选页

这里模拟每 128 步重新做一次：

```text
router + scoring + top-k + gather/compact
```

1024 步内一共 8 次重选。

k2 结果：

| full length | method | reroutes | overhead | total | speedup | overhead share |
|---:|---|---:|---:|---:|---:|---:|
| 4096 | k2_once | 1 | 2.439 ms | 6627.963 ms | 1.41x | 0.04% |
| 4096 | k2_interval128 | 8 | 19.513 ms | 5986.007 ms | 1.57x | 0.33% |
| 8192 | k2_once | 1 | 2.471 ms | 6733.509 ms | 2.58x | 0.04% |
| 8192 | k2_interval128 | 8 | 19.767 ms | 6008.204 ms | 2.89x | 0.33% |
| 16384 | k2_once | 1 | 2.431 ms | 6753.411 ms | 5.77x | 0.04% |
| 16384 | k2_interval128 | 8 | 19.448 ms | 6020.456 ms | 6.48x | 0.32% |
| 20000 | k2_once | 1 | 2.457 ms | 6803.479 ms | 6.95x | 0.04% |
| 20000 | k2_interval128 | 8 | 19.653 ms | 6059.496 ms | 7.80x | 0.32% |
| 32768 | k2_once | 1 | 2.444 ms | 6808.012 ms | 11.21x | 0.04% |
| 32768 | k2_interval128 | 8 | 19.554 ms | 6066.903 ms | 12.58x | 0.32% |

注意：

```text
interval128 在这个模拟里每 128 步会重建 compact KV，
所以 active KV 的 decode 增长被周期性重置；
因此它不只是多了 overhead，也可能减少后续 attention 的有效 KV 长度。
```

关键点是：

```text
即使 1024 步里重选 8 次，
overhead share 也只有约 0.3%。
```

## 总结

这次测试支持下面结论：

```text
1. attention/KV 子系统里，当前 recent-plus 方法速度是够的。
2. 额外引入的 router/scoring/top-k/gather/compact 开销很小。
3. 20k 长度下，k2/k3/k4 分别约 6.95x / 5.43x / 4.47x。
4. 32k 长度下，k2/k3/k4 分别约 11.21x / 8.75x / 7.16x。
5. 短上下文 4k 不适合激进报告速度，因为 active KV ratio 太高，k3/k4 收益很小。
```

论文里建议这样报告：

```text
Attention/KV subsystem speed:
  include router + scoring + top-k + gather/compact + compact attention

End-to-end generation speed:
  单独报告，说明未做完整 serving/kernel 集成时，MLP/lm_head/HF 调度会稀释收益。
```
