# Section 104: Block-size-aware runtime router 训练与初步评测

日期：2026-07-07

## 目的

前一轮 sweep 说明固定 `block=512, topK=3` 已经比 `block=1024` 更好，但不同任务的最优 block 数量和 block size 明显不同：

- 短上下文需要更保守，过小 active KV 容易不稳。
- 长上下文可以更激进，`block=256` 往往能把 token ratio 压到 5%-15%。
- 有些 LongBench / 多证据任务需要更多 topK 或更大 block。

因此这一步把 oracle sweep label 蒸馏成一个可推理时使用的小 router，让 router 同时选择：

- block size：`256 / 512 / 1024 / 2048`
- retrieval budget：`top1 / top2 / top3 / top4 / top6 / top8`

目前 action 形如：

```text
recent_plus_b256_span_top1_b0_a0
recent_plus_b256_span_top2_b0_a0
recent_plus_b512_span_top1_b0_a0
...
```

推理时 `recent raw` 仍然必选；router 只决定 old context 用多小的 block、取多少个 evidence block。

## 代码改动

新增蒸馏脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_blocksize_router_distill_from_sweeps.py
```

新增训练入口：

```text
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_blocksize_router_from_sweeps_qwen8b.sh
```

新增评测入口：

```text
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_blocksize_router_m3.sh
```

修改 Qwen3-8B benchmark runner：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

关键支持：

- 新增 `router_blocksize` method。
- 支持 router 输出 `recent_plus_b{block}_...`。
- `build_memory_for_action` 会根据 action 临时覆盖 `block_tokens`，因此同一次 benchmark 里可以混用不同 block size。

## 训练数据

router label 来自已有 Qwen3-8B m3 block/topK sweep：

```text
block = 256 / 512 / 1024 / 2048
topK  = 1 / 2 / 3 / 4 / 6 / 8
benchmark = LongBench + RULER 4k/8k/16k
max_examples_per_task = 3
```

oracle label 的选择规则：

- summary 类任务允许 `full_raw score - 0.03` 的 ROUGE-L slack。
- exact/retrieval 类任务优先要求达到 best/full threshold。
- 在满足质量阈值的候选中选择 token ratio 最低的 action。
- 如果某些 sweep 行分数异常或非有限值，会被过滤。
- 如果某个 case 没有压缩候选，保留为 `full_raw` 保守标签。

注意：这是从已有 sweep 上做的 router smoke 蒸馏，不是最终 held-out 训练。

## Router 蒸馏结果

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/blocksize_router_from_sweeps_m3_20260707
```

训练结果：

| split | samples | label accuracy | avg score | avg token ratio |
|---|---:|---:|---:|---:|
| train | 78 | 91.03% | 0.8557 | 15.14% |
| test | 18 | 72.22% | 0.8437 | 16.08% |

测试集 label accuracy 不算特别高，但多数错误是从 oracle 的一个低成本正确 action 预测到另一个仍然正确的低成本 action。对这类任务，最终更应该看质量和 token ratio，而不是只看 label exact match。

## 实际 Qwen3-8B m3 生成评测

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_blocksize_router_m3_20260707
```

整体结果：

| method | avg score | token ratio | avg seconds | speedup |
|---|---:|---:|---:|---:|
| full_raw | 0.8116 | 100.00% | 7.505s | 1.00x |
| recent_plus_span_top3_b0_a0 | 0.8107 | 18.22% | 4.496s | 1.67x |
| router_blocksize | 0.8535 | 11.04% | 4.274s | 1.76x |

分组结果：

| benchmark | full_raw score | router score | relative | router token ratio |
|---|---:|---:|---:|---:|
| LongBench | 0.2465 | 0.4140 | 167.97% | 16.85% |
| RULER 4k | 1.0000 | 1.0000 | 100.00% | 24.76% |
| RULER 8k | 1.0000 | 1.0000 | 100.00% | 13.13% |
| RULER 16k | 1.0000 | 1.0000 | 100.00% | 6.54% |

router 选择分布：

| action | count |
|---|---:|
| `recent_plus_b256_span_top1_b0_a0` | 27 |
| `recent_plus_b256_span_top2_b0_a0` | 19 |
| `recent_plus_b256_span_top4_b0_a0` | 10 |
| `recent_plus_b256_span_top6_b0_a0` | 10 |
| `recent_plus_b512_span_top1_b0_a0` | 9 |
| `recent_plus_b256_span_top3_b0_a0` | 9 |
| `recent_plus_b256_span_top8_b0_a0` | 5 |
| `recent_plus_b1024_span_top1_b0_a0` | 2 |
| `recent_plus_b2048_span_top1_b0_a0` | 2 |
| `recent_plus_b512_span_top2_b0_a0` | 2 |
| `recent_plus_b2048_span_top2_b0_a0` | 1 |

## 初步结论

`block-size-aware router` 的方向是对的：

- 相比固定 `block512 top3`，router 同时提高质量和降低 token ratio。
- RULER 16k 能保持 100% score，同时只用 6.54% token。
- LongBench 在这组 m3 样本上高于 full_raw，说明 evidence block routing 对部分 QA 类任务甚至有去噪效果。
- 实际端到端 speedup 只有 1.76x，低于理论 attention/KV 子模块速度，因为当前实现仍是 prompt 重组/普通 prefill 口径，不是 KV-native CUDA kernel 级实现。

## 风险与下一步

当前数字不能直接作为 paper final result：

- m3 样本太小。
- router label 来自同一批 sweep，虽然评测重新跑了生成，但还不是严格 held-out。
- 当前 runner 仍以重构 prompt 为主，KV-native gather 只是 smoke demo 级别。

下一步建议：

1. 用非 benchmark synthetic + book QA + held-out long text 训练 router，避免 benchmark 泄漏。
2. 在 LongBench/RULER full 或更大 m10/m50 上评测 `router_blocksize`。
3. 把 action space 收敛到更稳定的集合：`b256 top1/2/3/4/6/8`、`b512 top1/2/3`、少量 `b1024/b2048 top1/2`。
4. 做 KV-native gather 版 runtime benchmark，把 router/page scoring/gather/compact attention 都计入 attention/KV subsystem。
