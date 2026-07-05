# Section 64: attention-only 分页开销与新 prompt 均摊测试

## 目标

前面的 warm-cache 1k decode 端到端测速混入了 MLP、lm_head、HF forward、Python decode loop 等共同开销。那些不是本方法新引入的开销，也不是本方法主要优化的对象。

本节改用更合理的口径：

```text
只比较 attention/KV 子系统。

full_raw:
  多步 attention 到完整 KV。

page/summary memory:
  新 prompt 到来时，先做 router / page scoring / top-k / KV page gather。
  后续多步 attention 到较短 compact KV。

统计：
  新 prompt 一次性开销
  多步 attention 时间
  一次性开销摊到 1/16/64/256/1024 步后的总速度
```

脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_attention_paging_amortized_timing.py
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/attention_paging_amortized_20k_qwen06b_20260705
```

## 实验设置

使用 Qwen3-0.6B 的真实 attention 结构：

```text
layers = 28
heads = 16
head_dim = 128
dtype = fp16
```

full KV 长度：

```text
19455 tokens
```

page size：

```text
1024 tokens
```

测试的压缩/page 选择策略：

```text
page_once_2p: 新 prompt 来时选 2 个 page，后续一直复用
page_once_4p: 新 prompt 来时选 4 个 page，后续一直复用
page_once_5p: 新 prompt 来时选 5 个 page，后续一直复用
page_interval_128: 每 128 步重新做一次 router/top-k/gather
page_interval_1024: 每 1024 步重新做一次 router/top-k/gather
```

这里的 overhead 包含：

- tiny router MLP forward
- page scoring
- top-k
- across-layer K/V page gather
- K/V copy 到 compact buffer

不包含：

- MLP
- lm_head
- tokenizer
- HF generate
- 完整模型 Python forward 调度

## 关键结果：2 pages

2 pages 对应 active KV 长度约 `2048 tokens`。

| 方法 | steps | new prompt overhead | attention time | total time | speedup vs full | overhead share |
|---|---:|---:|---:|---:|---:|---:|
| full_attention | 1 | 0.000 ms | 5.636 ms | 5.636 ms | 1.00x | 0.00% |
| page_once_2p | 1 | 3.145 ms | 1.833 ms | 4.978 ms | 1.13x | 63.18% |
| full_attention | 16 | 0.000 ms | 88.565 ms | 88.565 ms | 1.00x | 0.00% |
| page_once_2p | 16 | 3.145 ms | 25.831 ms | 28.976 ms | 3.06x | 10.85% |
| full_attention | 64 | 0.000 ms | 353.695 ms | 353.695 ms | 1.00x | 0.00% |
| page_once_2p | 64 | 3.145 ms | 103.085 ms | 106.230 ms | 3.33x | 2.96% |
| full_attention | 256 | 0.000 ms | 1420.707 ms | 1420.707 ms | 1.00x | 0.00% |
| page_once_2p | 256 | 3.145 ms | 412.136 ms | 415.281 ms | 3.42x | 0.76% |
| full_attention | 1024 | 0.000 ms | 5774.912 ms | 5774.912 ms | 1.00x | 0.00% |
| page_once_2p | 1024 | 3.145 ms | 1649.477 ms | 1652.621 ms | 3.49x | 0.19% |

结论：

```text
新 prompt 的 page selection/gather 一次性开销大约是 3.1 ms。
如果只生成 1 token，这个开销非常明显。
但摊到 64/256/1024 步后，它基本不是瓶颈。
```

## 间隔压缩/重选页

2 pages 下，比较 `page_once`、`每 128 步重选`、`每 1024 步重选`：

| 方法 | steps | overhead | attention | total | speedup |
|---|---:|---:|---:|---:|---:|
| page_once_2p | 1024 | 3.145 ms | 1649.477 ms | 1652.621 ms | 3.49x |
| page_interval_128_2p | 1024 | 25.158 ms | 1650.915 ms | 1676.073 ms | 3.45x |
| page_interval_1024_2p | 1024 | 3.145 ms | 1652.824 ms | 1655.969 ms | 3.49x |

这说明：

- 每 128 步重新 router/top-k/gather 一次，1024 步里要做 8 次，开销变成约 25 ms。
- 即使这样，overhead share 也只有约 1.5%。
- 每 1024 步重选一次几乎等同于 page_once。

因此，你说的“不要每一步压缩，应该间隔一定步数再压缩”是正确的；在 attention-only 口径下，128 或 1024 步 interval 的开销都可以被很好摊薄。

## 不同 page 数量

1024 步时：

| 方法 | active KV | overhead | attention | total | speedup | overhead share |
|---|---:|---:|---:|---:|---:|---:|
| full_attention | 19455 | 0.000 ms | 5774.912 ms | 5774.912 ms | 1.00x | 0.00% |
| page_once_2p | 2048 | 3.145 ms | 1649.477 ms | 1652.621 ms | 3.49x | 0.19% |
| page_once_4p | 4096 | 6.228 ms | 1655.920 ms | 1662.148 ms | 3.47x | 0.37% |
| page_once_5p | 5120 | 7.790 ms | 1878.746 ms | 1886.536 ms | 3.06x | 0.41% |

这里可以看到 page 数增加后，一次性 gather 开销线性增加，但在 1024 步里仍然很小。主要差别来自 attention active KV 长度。

## 解释

这个测试比前面的 HF warm decode 更接近我们真正想要优化和报告的东西：

```text
只看 attention/KV 子系统；
把本方法新增的 router/page scoring/top-k/gather 算进去；
不把 MLP/lm_head 等共同成本算进来。
```

在这个口径下，结论是积极的：

```text
20k full KV -> 2k paged KV:
  新 prompt overhead: 约 3.1 ms
  16-step amortized speedup: 3.06x
  64-step amortized speedup: 3.33x
  256-step amortized speedup: 3.42x
  1024-step amortized speedup: 3.49x
```

也就是说，本方法真正优化的 attention/KV 部分是能加速的，而且 page selection/gather 的新增成本可以通过多步 decode 很快摊薄。

## 对论文实验的建议

后续应该至少报告三张表：

1. `Prefill/TTFT`：说明长 prompt 重新进入模型时的速度。
2. `Attention/KV subsystem`：像本节这样，报告 router/top-k/gather + 多步 attention 的摊销速度。
3. `End-to-end serving`：说明在未做 kernel/serving 优化时，完整 HF decode 还不能释放这些收益。

这样表述更严谨，也更符合旧 KV retrieval 论文常用的 kernel/subsystem 加速口径。
